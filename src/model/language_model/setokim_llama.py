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


from cProfile import label
from typing import List, Optional, Tuple, Union, Callable
import math
import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import ModelOutput
from ..setokim_arch import SetokimMetaModel, SetokimMetaForCausalLM

def mask_by_order(mask_len, order, bsz, seq_len):
    masking = torch.zeros(bsz, seq_len).cuda()
    masking = torch.scatter(masking, dim=-1, index=order[:, :mask_len.long()], src=torch.ones(bsz, seq_len).cuda()).bool()
    return masking

class SetokimConfig(LlamaConfig):
    model_type = "setokim_llama"


class SetokimLlamaModel(SetokimMetaModel, LlamaModel):
    config_class = SetokimConfig

    def __init__(self, config: LlamaConfig):
        super(SetokimLlamaModel, self).__init__(config)


class SetokimLlamaForCausalLM(LlamaForCausalLM, SetokimMetaForCausalLM):
    config_class = SetokimConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = SetokimLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
        
    def get_model(self):
        return self.model

    def get_lm_head(self):
        return self.lm_head

    def sample_orders(self, bsz):
        # generate a batch of random generation orders
        orders = []
        for _ in range(bsz):
            order = np.array(list(range(self.seq_len)))
            np.random.shuffle(order)
            orders.append(order)
        orders = torch.Tensor(np.array(orders)).long()
        return orders

    def random_masking(self, x, orders):
        # generate token mask
        bsz, seq_len, embed_dim = x.shape
        mask_ratio_generator = stats.truncnorm((self.config.mask_ratio_min - 1.0) / 0.25, 0, loc=1.0, scale=0.25)
        mask_rate = mask_ratio_generator.rvs(1)[0]
        num_masked_tokens = int(np.ceil(seq_len * mask_rate))
        mask = torch.zeros(bsz, seq_len, device=x.device)
        mask = torch.scatter(mask, dim=-1, index=orders[:, :num_masked_tokens],
                             src=torch.ones(bsz, seq_len, device=x.device, dtype=x.dtype))
        return mask
    
    def compute_diff_loss(self, z, target, mask):
        # bsz, seq_len, _ = target.shape
        target = target.repeat(self.config.diffusion_batch_mul, 1)
        z = z.repeat(self.config.diffusion_batch_mul, 1)
        mask = mask.repeat(self.config.diffusion_batch_mul)
        loss = self.get_diffloss()(z=z, target=target, mask=mask)
        return loss

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        comp_images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        gen_images: Optional[torch.FloatTensor] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                new_labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                comp_images,
                image_sizes
            )

        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]

        # 只计算部分的
        logits = self.lm_head(hidden_states)

        loss = None
        if new_labels is not None:
            logits = logits.float()
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                shift_attention_mask = attention_mask[..., 1:]
                shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = new_labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = new_labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
            )

        diff_loss_list = []
        if gen_images is not None:
            # target_token_hidden_states = hidden_states[labels==self.config.target_token_index]
            # bsz = hidden_states.size(0)
            # mask_list = p
            # for i in range(bsz):
            #     orders = self.sample_orders(bsz=target_token_hidden_states.size(0)).to(target_token_hidden_states.device)
            #     mask = self.random_masking(target_token_hidden_states, orders)
            # diff_loss = self.compute_diff_loss(target_token_hidden_states, gen_images, mask)
            for i in range(hidden_states.size(0)):
                target_token_hidden_states = hidden_states[i][labels[i]==self.config.target_token_index]
                target_token_hidden_states = target_token_hidden_states.unsqueeze(0)

                orders = self.sample_orders(bsz=target_token_hidden_states.size(0)).to(target_token_hidden_states.device)
                mask = self.random_masking(target_token_hidden_states, orders)
                diff_loss = self.compute_diff_loss(target_token_hidden_states, gen_images[i].unsqueeze(0), mask)
                diff_loss_list.append(diff_loss)
        
            loss += sum(diff_loss_list)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        
    def unpatchify(self, x):
        bsz = x.shape[0]
        p = self.config.vision_generator_config.patch_size
        c = self.config.vision_generator_config.decoder_embed_dim
        image_size = self.config.vision_generator_config.image_size
        h_, w_ = image_size // p, image_size // p

        x = x.reshape(bsz, h_, w_, c, p, p)
        x = torch.einsum('nhwcpq->nchpwq', x)
        x = x.reshape(bsz, c, h_ * p, w_ * p)
        return x  # [n, c, h, w]
    
    def sample_tokens(self, x, seq_len=64, num_iter=64, cfg=1.0, cfg_schedule="linear", temperature=1.0, progress=False):

        # init and sample generation orders
        bsz = x.shape[0]
        mask = torch.ones(bsz, seq_len).cuda()
        tokens = torch.zeros(bsz, seq_len, self.config.hidden_size).cuda()
        orders = self.sample_orders(bsz)

        indices = list(range(num_iter))
        if progress:
            indices = tqdm(indices)
        # generate latents
        for step in indices:
            cur_tokens = tokens.clone()
            
            tokens = torch.cat([tokens, tokens], dim=0)
            mask = torch.cat([mask, mask], dim=0)

            # mae decoder
            z = x

            # mask ratio for the next round, following MaskGIT and MAGE.
            mask_ratio = np.cos(math.pi / 2. * (step + 1) / num_iter)
            mask_len = torch.Tensor([np.floor(seq_len * mask_ratio)]).cuda()

            # masks out at least one for the next iteration
            mask_len = torch.maximum(torch.Tensor([1]).cuda(),
                                     torch.minimum(torch.sum(mask, dim=-1, keepdims=True) - 1, mask_len))

            # get masking for next iteration and locations to be predicted in this iteration
            mask_next = mask_by_order(mask_len[0], orders, bsz, seq_len)
            if step >= num_iter - 1:
                mask_to_pred = mask[:bsz].bool()
            else:
                mask_to_pred = torch.logical_xor(mask[:bsz].bool(), mask_next.bool())
            mask = mask_next
            if not cfg == 1.0:
                mask_to_pred = torch.cat([mask_to_pred, mask_to_pred], dim=0)

            # sample token latents for this step
            z = z[mask_to_pred.nonzero(as_tuple=True)]
            # cfg schedule follow Muse
            if cfg_schedule == "linear":
                cfg_iter = 1 + (cfg - 1) * (seq_len - mask_len[0]) / seq_len
            elif cfg_schedule == "constant":
                cfg_iter = cfg
            else:
                raise NotImplementedError
            sampled_token_latent = self.get_diffloss().sample(z, temperature, cfg_iter)
            if not cfg == 1.0:
                sampled_token_latent, _ = sampled_token_latent.chunk(2, dim=0)  # Remove null class samples
                mask_to_pred, _ = mask_to_pred.chunk(2, dim=0)

            cur_tokens[mask_to_pred.nonzero(as_tuple=True)] = sampled_token_latent
            tokens = cur_tokens.clone()

        # unpatchify
        tokens = self.unpatchify(tokens)
        return tokens
    
    @torch.no_grad()
    def _get_generation(
        self, 
        input_ids: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        max_new_tokens: Optional[int] = 200,
        top_p: Optional[float] = 10.0,
        temperature: Optional[float] = 0.1,
        stopping_criteria: Optional[Callable] = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs):
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                input_ids,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(input_ids)

        if attention_mask is None:
            attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.bool, device=inputs_embeds.device)

            batch_size, seq_length = attention_mask.shape
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device).expand((batch_size, seq_length))
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        outputs = super().generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            temperature=temperature,
            stopping_criteria=stopping_criteria,
            output_hidden_states=output_hidden_states,
            return_dict_in_generate=True,
            output_attentions=True,
            **kwargs
        )
        return outputs
    

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        output_attentions = kwargs.pop("output_attentions", True)
        output_hidden_states = kwargs.pop("output_hidden_states", True)
        max_new_tokens = kwargs.pop("max_new_tokens", 200)
        top_p = kwargs.pop("top_p", 10.0)
        temperature = kwargs.pop("temperature", 0.1)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        outputs = self._get_generation(
            input_ids=inputs, 
            images=images,
            image_sizes=image_sizes,
            position_ids=position_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            temperature=temperature
        )

        generated_ids = outputs.sequences
        # the output hidden states is a tuple - 
        # one is the input hidden states of all layers (32 + 1(embedding layers)) 
        # and the other is the output hidden states of all layers (32 + 1(embedding layers))
        hidden_embedding = [x[-1] for x in outputs.hidden_states[1:]]
        print('hidden_embedding: ', len(hidden_embedding))
        print('hidden_embedding: ', hidden_embedding[0].size())
        hidden_embedding = torch.cat(hidden_embedding, dim=1)
        
        bsz = hidden_embedding.size(0)
        batch_image_outputs = []
        for i in range(bsz):
            start_pos = (generated_ids[i] == self.config.image_start_index).nonzero(as_tuple=False).tolist()
            end_pos = (generated_ids[i] == self.config.image_end_index).nonzero(as_tuple=False).tolist()
            _temp = []
            for s, e in zip(start_pos, end_pos):
                assert e[0] == s[0], (s, e)
                target_token_hidden_states = hidden_embedding[i][s[1]:e[1] + 1]
                image_output = self.sample_tokens(target_token_hidden_states)
                _temp.append(image_output)
            batch_image_outputs.append(_temp)
        return {
            outputs
        }


    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

AutoConfig.register("setokim_llama", SetokimConfig)
AutoModelForCausalLM.register(SetokimConfig, SetokimLlamaForCausalLM)
