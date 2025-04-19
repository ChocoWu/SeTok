import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Dict, List, Tuple, Optional
from einops import rearrange, repeat
from timm.models.vision_transformer import Block
from torch.utils.checkpoint import checkpoint
from transformers.models.bert import BertConfig
from .module import BertModel, PositionalEncoding2D
from diffusers.models.autoencoders.vae import Decoder



class SetokDeTokenizer(nn.Module):
    def __init__(self, 
                 token_feat_dim: Optional[int] = 4096,
                 hidden_dim: Optional[int] = 4096,
                 patch_size: Optional[int]=14, 
                 image_size: Optional[int]=256, 
                 decoder_embed_dim: Optional[int]=4096,
                 decoder_nheads: Optional[int]=16,
                 proj_drop: Optional[float]=0.2, 
                 attn_drop: Optional[float]=0.2,
                 decoder_depth: Optional[int]=16,
                 norm_layer: nn.Module = nn.LayerNorm,
                 mlp_ratio: Optional[float]=4.0,
                 feature_mapper_path_or_name: Optional[str]="bert-base-uncased",
                 num_hidden_layers: Optional[int]=6,
                 cross_attention_freq: Optional[int]=2,
                 initializer_range: Optional[float]=0.02,
                 **kwargs) -> None:
        super().__init__()
        self.token_feat_dim = token_feat_dim

        self.patch_size = patch_size
        self.height = self.weight = image_size // patch_size
        self.num_mask_token = self.height * self.weight
        self.hidden_dim = hidden_dim
        
        query_tokens = nn.Parameter(torch.zeros(1, self.num_mask_token, self.hidden_dim))
        query_tokens.data.normal_(mean=0.0, std=initializer_range)
        self.mask_tokens = query_tokens
        
        self.decoder_embed_dim = decoder_embed_dim
        self.mapper_fc_in = nn.Linear(self.token_feat_dim, self.hidden_dim)
        self.decoder_fc_in = nn.Linear(self.hidden_dim, self.decoder_embed_dim)
        
        self.decoder_norm = norm_layer(self.decoder_embed_dim)
        self.pixel_decoder = nn.ModuleList([
            Block(self.decoder_embed_dim, decoder_nheads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, proj_drop=proj_drop, attn_drop=attn_drop) for _ in range(decoder_depth)
        ])
        self.position_embedding = PositionalEncoding2D(self.hidden_dim)
        self.initialize_weights()
        self.init_feature_mapper(feature_mapper_path_or_name, self.hidden_dim, self.num_mask_token, num_hidden_layers, cross_attention_freq)

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

    def init_feature_mapper(
        self,
        feature_mapper_path_or_name: str,
        vision_width: int,
        num_mask_token: int,
        num_hidden_layers: int,
        cross_attention_freq: int
    ):
        print("feature_mapper_path_or_name: ", feature_mapper_path_or_name)
        mapper_config = BertConfig.from_pretrained(feature_mapper_path_or_name)

        mapper_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        mapper_config.add_cross_attention = True
        
        mapper_config.cross_attention_freq = cross_attention_freq
        mapper_config.query_length = num_mask_token
        mapper_config.num_hidden_layers = num_hidden_layers

        self.mapper = BertModel.from_pretrained(feature_mapper_path_or_name, config=mapper_config)
        self.mapper.cls = None
        self.mapper.embeddings.word_embeddings = None
        self.mapper.embeddings.position_embeddings = None
        for layer in self.mapper.encoder.layer:
            layer.output = None
            layer.intermediate = None

    def load_model(self):
        pass

    def forward(self, x, attention_masks):

        mask_tokens = self.mask_tokens.expand(x.shape[0], -1, -1)
        x = self.mapper_fc_in(x)
        x = self.mapper(
            query_embeds=mask_tokens,
            encoder_hidden_states=x,
            encoder_attention_mask=attention_masks,
            return_dict=True).last_hidden_state

        x = self.decoder_fc_in(x)  # b, h*w, c
        _x = rearrange(x, 'B (h w) C -> B h w C', h=self.height, w=self.weight)
        pos_emb = self.position_embedding(_x)
        pos_emb = rearrange(pos_emb, 'B h w C -> B (h w) C')
        x = x + pos_emb

        for block in self.pixel_decoder:
            x = block(x)
        
        x = self.decoder_norm(x)

        
        
        