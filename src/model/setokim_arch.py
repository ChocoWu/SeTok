#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import torch
from dataclasses import asdict

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector
from .multimodal_generator.builder import build_vision_generator
from .loss import DiffLoss
from src.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, TARGET_TOKEN_INDEX
from src.constants import DEFAULT_TARGET_TOKEN



class SetokimMetaModel:

    def __init__(self, config):
        super(SetokimMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config.vision_tower_config, delay_load=True)
            self.mm_in_projector = build_vision_projector(config.vision_in_projector_config)
        
        if hasattr(config, "mm_vision_generator"):
            self.vision_generator = build_vision_generator(config.vision_generator_config)
            self.mm_out_projector = build_vision_projector(config.vision_out_projector_config)

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def get_vision_generator(self):
        vision_generator = getattr(self, 'vision_generator', None)
        if type(vision_generator) is list:
            vision_generator = vision_generator[0]
        return vision_generator
    
    def get_input_projector(self):
        mm_in_projector = getattr(self, 'mm_in_projector', None)
        if type(mm_in_projector) is list:
            mm_in_projector = mm_in_projector[0]
        return mm_in_projector

    def get_output_projector(self):
        mm_out_projector = getattr(self, 'mm_out_projector', None)
        if type(mm_out_projector) is list:
            mm_out_projector = mm_out_projector[0]
        return mm_out_projector

    def get_diffloss(self):
        diffloss = getattr(self, 'diffloss', None)
        if type(diffloss) is list:
            diffloss = diffloss[0]
        return diffloss
    
    def initialize_vision_modules(self, vision_tower_args, proj_in_args, fsdp=None):
        vision_tower = vision_tower_args.vision_tower

        self.config.mm_vision_tower = vision_tower
        self.config.vision_tower_config = asdict(vision_tower_args)
        self.config.token_feat_dim = vision_tower_args.token_feat_dim

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(vision_tower)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        if vision_tower_args.pretrain_vision_tokenizer is not None:
            pretrain_vision_tokenizer_weights = torch.load(vision_tower_args.pretrain_vision_tokenizer, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.vision_tower.load_state_dict(get_w(pretrain_vision_tokenizer_weights, 'tokenizer'), strict=False)

        self.config.use_mm_in_proj = True
        self.config.mm_in_projector_type = getattr(proj_in_args, 'mm_in_projector_type', 'linear')
        self.config.vision_in_projector_config = asdict(proj_in_args)
        pretrain_mm_in_mlp_adapter = proj_in_args.pretrain_mm_in_mlp_adapter

        if getattr(self, 'mm_in_projector', None) is None:
            self.mm_in_projector = build_vision_projector(
                getattr(proj_in_args, 'mm_in_projector_type', 'linear'), 
                getattr(proj_in_args, 'mm_hidden_size', 4096), 
                getattr(proj_in_args, 'hidden_size', 3078)
            )
        else:
            # In case it is frozen by LoRA
            for p in self.mm_in_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_in_mlp_adapter is not None:
            mm_in_projector_weights = torch.load(pretrain_mm_in_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_in_projector.load_state_dict(get_w(mm_in_projector_weights, 'mm_in_projector'), strict=False)
    
    def initialize_vision_generator_modules(self, vision_generator_args, proj_out_args, fsdp=None):

        self.config.vision_generator_config = asdict(vision_generator_args)

        if self.get_vision_generator() is None:
            vision_generator = build_vision_generator(vision_generator_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_generator = [vision_generator]
            else:
                self.vision_generator = vision_generator
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_generator = self.vision_generator[0]
            else:
                vision_generator = self.vision_generator
            vision_generator.load_model()

        if vision_generator_args.pretrain_vision_detokenizer is not None:
            pretrain_vision_tokenizer_weights = torch.load(vision_generator_args.pretrain_vision_detokenizer, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.vision_generator.load_state_dict(get_w(pretrain_vision_tokenizer_weights, 'detokenizer'), strict=False)

        self.config.use_mm_out_proj = True
        self.config.mm_out_projector_type = getattr(proj_out_args, 'mm_out_projector_type', 'linear')
        self.config.vision_out_projector_config = asdict(proj_out_args)
        pretrain_mm_out_mlp_adapter = proj_out_args.pretrain_mm_out_mlp_adapter

        if getattr(self, 'mm_out_projector', None) is None:
            self.mm_out_projector = build_vision_projector(
                getattr(proj_out_args, 'mm_out_projector_type', 'linear'), 
                getattr(proj_out_args, 'mm_hidden_size', 4096), 
                getattr(proj_out_args, 'hidden_size', 3078)
            )
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_out_mlp_adapter is not None:
            mm_out_projector_weights = torch.load(pretrain_mm_out_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_out_projector.load_state_dict(get_w(mm_out_projector_weights, 'mm_out_projector'), strict=False)

    def initialize_diffloss(self, diff_loss_args):
        self.diffloss = DiffLoss(
            target_channels=self.config.token_feat_dim,
            z_channels=self.config.hidden_size,
            width=diff_loss_args.diffloss_w,
            depth=diff_loss_args.diffloss_d,
            num_sampling_steps=diff_loss_args.num_sampling_steps,
            grad_checkpointing=diff_loss_args.grad_checkpointing
        )
        self.config.diffusion_batch_mul = diff_loss_args.diffusion_batch_mul
        self.config.mask_ratio_min= diff_loss_args.mask_ratio_min
        self.config.diff_loss_config = asdict(diff_loss_args)

class SetokimMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def get_input_projector(self):
        return self.get_model().get_input_projector()
    
    def get_vision_generator(self):
        return self.get_model().get_vision_generator()

    def get_output_projector(self):
        return self.get_model().get_output_projector()

    def get_diffloss(self):
        return self.get_model().get_diffloss()

    def encode_images(self, images):
        image_features, _, _ = self.get_model().get_vision_tower()(images)
        if image_features.dim() == 4:
            image_features = image_features.flatten(2).transpose(1, 2)  # [b, h*w, c] 
        image_features = self.get_model().mm_in_projector(image_features)
        return image_features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if type(images) is list or images.ndim == 5:
            concat_images = torch.stack([image for image in images], dim=0)
            # print('images: ', concat_images.shape)  # 16, 3, 448, 448
            image_features = self.encode_images(concat_images)
        else:
            image_features = self.encode_images(images)

        # TODO: image start / end is not implemented here to support pretraining.
        # if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
        #     raise NotImplementedError

        # new_tmp = []
        # cluster_num = []
        # idx_cluster_list = []
        # for image_feature_i in image_features:
        #     _temp, idx_cluster = self.get_model().mm_projector(image_feature_i, self.config.k, self.config.threshold)
        #     new_tmp.append(_temp)
        #     cluster_num.append(_temp.shape[0])
        #     idx_cluster_list.append(idx_cluster.to('cpu'))
        # image_features = new_tmp

        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded
            new_labels[new_labels == TARGET_TOKEN_INDEX] = IGNORE_INDEX

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_TARGET_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_in_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_in_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_in_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_in_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
