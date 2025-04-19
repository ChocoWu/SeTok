import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Dict, List, Optional
import numpy as np
from einops import rearrange, repeat
from timm.models.layers import DropPath
import math
from .clip_encoder import CLIPVisionTower
from .module import Block, PositionalEncoding2D


class SetokTokenizer(nn.Module):
    def __init__(self, 
                 vision_tower: str = 'google/siglip-so400m-patch14-384', 
                 unfreeze_mm_vision_tower: Optional[bool] = False, 
                 mm_vision_select_feature: Optional[str] = 'patch',
                 mm_vision_select_layer: Optional[int] = -2,
                 delay_load: Optional[bool]= False,
                 hidden_dim: Optional[int] = 4096,
                 token_feat_dim: Optional[int] = 4096,
                 min_cluster_num: Optional[int] = 64,
                 threshold: Optional[float] = 0.5,
                 nheads: Optional[int] = 2, 
                 dim_feedforward: Optional[int] = 4096, 
                 proj_drop: Optional[float] = 0.2, 
                 drop_path: Optional[float] = 0.0, 
                 inner_cluster_layers: Optional[int] = 2,
                 intra_cluster_layers: Optional[int] = 2,
                 attn_drop: Optional[float] = 0.0,
                 act_layer: nn.Module = nn.GELU,
                 norm_layer: nn.Module = nn.LayerNorm,
                 **kwargs
                 ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.token_feat_dim = token_feat_dim

        self.inner_encoder = Block(self.hidden_dim, nheads, dim_feedforward, proj_drop=proj_drop, attn_drop=attn_drop, drop_path=drop_path, act_layer=act_layer, norm_layer=norm_layer, depth=inner_cluster_layers)
        self.inter_encoder = Block(self.hidden_dim, nheads, dim_feedforward, proj_drop=proj_drop, attn_drop=attn_drop, drop_path=drop_path, act_layer=act_layer, norm_layer=norm_layer, depth=intra_cluster_layers)
        self.position_embedding = PositionalEncoding2D(self.hidden_dim)
        self.out = nn.Linear(self.hidden_dim, self.token_feat_dim)

        self.min_cluster_num = min_cluster_num
        self.threshold = threshold

        self.initialize_weights()

        self.image_feature_encoder = CLIPVisionTower(vision_tower, 
                 unfreeze_mm_vision_tower=unfreeze_mm_vision_tower, 
                 mm_vision_select_feature=mm_vision_select_feature, 
                 mm_vision_select_layer=mm_vision_select_layer, 
                 delay_load=delay_load
        )
        self.image_processor = self.image_feature_encoder.image_processor
        

    def initialize_weights(self):
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    @property
    def dtype(self):
        return self.Linear.weight.dtype
    
    def cluster_dpc_knn(self, x, k, token_mask=None, threshold=0.53):
        with torch.no_grad():
            N, C = x.shape

            dist_matrix = torch.cdist(x, x) / (C ** 0.5)  # C * C

            if token_mask is not None:
                token_mask = token_mask > 0
                dist_matrix = dist_matrix * token_mask[None, :] + (dist_matrix.max() + 1) * (~token_mask[None, :])

            dist_nearest, index_nearest = torch.topk(dist_matrix, k=k, dim=-1, largest=False)  # C * k

            density = (-(dist_nearest ** 2).mean(dim=-1)).exp()  # C
            density = density + torch.rand(density.shape, device=density.device, dtype=density.dtype) * 1e-6  # C

            if token_mask is not None:
                density = density * token_mask

            mask = density[None, :] > density[:, None]  # C * C
            mask = mask.type(x.dtype)
            dist_max = dist_matrix.flatten(1).max(dim=-1)[0][None, None]  # C * C
            dist, index_parent = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)  # 1 * C, 1 * C

            score = dist * density

            index_down = torch.nonzero(score.reshape(-1)>threshold).reshape(-1)  # obtain the index of the center
            if index_down.numel() == 0:
                _, index_down = torch.topk(score, k=self.min_cluster_num, dim=-1) 
                index_down = torch.sort(index_down).values
                index_down = index_down.reshape(-1)

            # obtain the index of the cluster that each token belongs to
            # dist_matrix = index_points(dist_matrix, index_down.squeeze())  # the cluster_num * C
            dist_matrix = dist_matrix[index_down, :]  # the cluster_num * C

            idx_cluster = dist_matrix.argmin(dim=0)  # the cluster_num
            
            # B = 1
            # idx_batch = torch.arange(B, device=x.device)[:, None].expand(cluster_num)
            cluster_num = index_down.size(0)
            idx_tmp = torch.arange(cluster_num, device=x.device)[None, :]
            idx_cluster[index_down] = idx_tmp.reshape(-1)

        return index_down, idx_cluster, score
    
    def group_encoding(self, x, centers, labels):
        """
        We apply transformer within each group to modeling the features.
        Specifically, we take the center representation as the initial representation of CLS. 
        Then, we take the CLS representation as the final representation of the group, i.e., the concept-level visual token.
        Args:
            x of size (W, C), 
            centers of size (L, C)
            label of size (W)

        Return: 
            Output: group features of size (L, C)
        """

        W, C = x.size()
        L, _ = centers.size()

        # Compute masks for each unique label
        unique_labels, label_counts = labels.unique(return_counts=True)
        # print('unique_labels: ', unique_labels) 
        masks = [labels == cur_label for cur_label in unique_labels]  # L, W
        # centers = centers.index_select(1, unique_labels)

        group_features = []
        for i, m in enumerate(masks):
            # _m = m.unsqueeze(1).expand(W, C)
            # cur_length = torch.sum(m).item()
            _cur_cluster_feat = self.inner_encoder(x[m].unsqueeze(0))
            _cur_cluster_feat = _cur_cluster_feat.squeeze(0).mean(dim=0)
            group_features.append(_cur_cluster_feat)
        group_features = torch.stack(group_features, dim=0)

        return group_features

    def forward(self, x, k=None, threshold=None, token_mask=None):
        """
        Expected Input: x of size (B, h, w, C)
        """ 
        x = self.image_feature_encoder(x)
        x = x.unsqueeze(0)
        B, hw, C = x.shape
        h = w = int(math.sqrt(x.shape[1]))
        _x = rearrange(x, 'B (h w) C -> B h w C', h=h, w=w)
        pos_emb = self.position_embedding(_x)
        pos_emb = rearrange(pos_emb, 'B h w C -> B (h w) C')
        x = x + pos_emb 
        x = x.squeeze(0) 
        
        _threshold = threshold if threshold else self.threshold
        _k = k if k else self.min_cluster_num

        index_down, idx_cluster, score = self.cluster_dpc_knn(x, _k, token_mask, _threshold)
        # index_down: the center index w.r.t the input x for each cluster
        # idx_cluster: the cluster index for each token in x, which is the index of the center in list [index_down]
        centers = x[index_down, :]
        group_features = self.group_encoding(x, centers, idx_cluster)
        group_features = self.inter_encoder(group_features)
        group_features = self.out(group_features)

        return group_features, idx_cluster, score