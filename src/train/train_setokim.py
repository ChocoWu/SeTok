# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
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

import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List
import torch
import transformers
import tokenizers
from torch.utils.data import Dataset
from src.model import *
from src.dataset import *
from setokim_trainer import SetokimTrainer
from training_utils import *



def rank0_print(*args):
    if local_rank == 0:
        print(*args)


from packaging import version
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        weight_to_save_buffer = get_mm_adapter_state_maybe_zero_3(trainer.model.named_buffers(), keys_to_match)
        weight_to_save.update(weight_to_save_buffer)
        
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg



def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args,
                                siglipTokenizer: Optional[transformers.PreTrainedTokenizer]=None,
                                visionTokenizer: Optional[torch.nn.Module]=None,
                                ) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    if data_args.task_type == 'pair':

        train_dataset = TextImagePairDataset(
            tokenizer=tokenizer,
            data_path=data_args.data_path,
            data_args=data_args,
            constrative_tokenizer=siglipTokenizer,
            vision_tokenizer=visionTokenizer
        )
        data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, siglipTokenizer=siglipTokenizer)
    elif data_args.task_type == 'instruction':
        train_dataset = InstructionTuningDataset(
            tokenizer=tokenizer,
            data_path=data_args.data_path,
            data_args=data_args
        )
        data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, siglipTokenizer=siglipTokenizer)
    elif data_args.task_type == 'edit':
        train_dataset = EditingDataset(
            tokenizer=tokenizer,
            data_path=data_args.data_path,
            data_args=data_args
        )
        data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, siglipTokenizer=siglipTokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def train(attn_implementation=None):
    global local_rank
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, VisionTowerArguments, VisionInProjectionArguments, VisionOutProjectionArguments, VisionGeneratorArguments, DiffLossArguments))
    model_args, data_args, training_args, vision_tower_args, vision_in_proj_args, vision_out_proj_args, vision_generator_args, diff_loss_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    print('model_args:', model_args)
    print('data_args: ', data_args)
    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))
    
    model = SetokimLlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=attn_implementation,
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        **bnb_model_from_pretrained_args
    )
    model.config.use_cache = False

    
    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    siglipTokenizer = transformers.AutoTokenizer.from_pretrained(
        vision_tower_args.vision_tower,
        cache_dir=training_args.cache_dir,
        padding_side="right",
        use_fast=False,
    )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        

    if model_args.vision_tokenizer is not None:
        model.get_model().initialize_vision_modules(
            vision_tower_args=vision_tower_args, 
            proj_in_args=vision_in_proj_args,
            fsdp=training_args.fsdp
        )
        
        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        model.config.tune_mm_in_mlp_adapter = training_args.tune_mm_in_mlp_adapter
        if training_args.tune_mm_in_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_in_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_in_mlp_adapter = training_args.freeze_mm_in_mlp_adapter
        if training_args.freeze_mm_in_mlp_adapter:
            for p in model.get_model().mm_in_projector.parameters():
                p.requires_grad = False

        if training_args.bits in [4, 8]:
            model.get_model().mm_in_projector.to(dtype=compute_dtype, device=training_args.device)
        model.get_input_projector().to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_in_projector_lr = training_args.mm_in_projector_lr
        model.config.mm_out_projector_lr = training_args.mm_out_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model_args.tune_mm_in_mlp_adapter = training_args.tune_mm_in_mlp_adapter
        model_args.pretrain_mm_in_mlp_adapter = vision_in_proj_args.pretrain_mm_in_mlp_adapter
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    if model_args.vision_generator is not None:
        vision_generator_args.token_feat_dim = vision_tower_args.token_feat_dim
        vision_generator_args.hidden_dim = 768
        vision_generator_args.proj_drop = 0.4
        vision_generator_args.attn_drop = 0.0
        vision_out_proj_args.mm_hidden_size = vision_in_proj_args.hidden_size
        vision_out_proj_args.hidden_size = vision_tower_args.hidden_dim

        model.get_model().initialize_vision_generator_modules(
            vision_generator_args=vision_generator_args, 
            proj_out_args=vision_out_proj_args,
            fsdp=training_args.fsdp
        )
        
        vision_generator = model.get_vision_generator()
        vision_generator.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        model.config.tune_mm_out_mlp_adapter = training_args.tune_mm_out_mlp_adapter
        if training_args.tune_mm_out_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_out_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_out_mlp_adapter = training_args.freeze_mm_out_mlp_adapter
        if training_args.freeze_mm_out_mlp_adapter:
            for p in model.get_model().mm_out_projector.parameters():
                p.requires_grad = False

        if training_args.bits in [4, 8]:
            model.get_model().mm_out_projector.to(dtype=compute_dtype, device=training_args.device)
        model.get_output_projector().to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

    if model_args.diff_loss is not None:
        model.get_model().initialize_diffloss(diff_loss_args)
        model.get_diffloss().to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

    model.config.image_start_index = tokenizer.tokenize('<im_start>')[0]
    model.config.image_end_index = tokenizer.tokenize('<im_end>')[0]

    rank0_print("model config \n: ", model.config)

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args,
                                              siglipTokenizer=siglipTokenizer,
                                              visionTokenizer=model.get_vision_tower())
    trainer = SetokimTrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
