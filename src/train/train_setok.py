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
from src.model import SeTok
from src.dataset import *
from setok_trainer import SetokTrainer
from training_utils import *



def rank0_print(*args):
    if local_rank == 0:
        print(*args)


from packaging import version
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    trainer.model.config.save_pretrained(output_dir)

    current_folder = output_dir.split('/')[-1]
    parent_folder = os.path.dirname(output_dir)
    if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
        if current_folder.startswith('checkpoint-'):
            mm_projector_folder = os.path.join(parent_folder, "mm_projector")
            os.makedirs(mm_projector_folder, exist_ok=True)
            torch.save(trainer.model.state_dict(), os.path.join(mm_projector_folder, f'{current_folder}.bin'))
        else:
            torch.save(trainer.model.state_dict(), os.path.join(output_dir, f'mm_projector.bin'))
    return


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args,
                                siglipTokenizer: Optional[transformers.PreTrainedTokenizer]=None,
                                ) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = TextImagePairDataset(
        tokenizer=tokenizer,
        data_path=data_args.data_path,
        data_args=data_args,
        constrative_tokenizer=siglipTokenizer
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, siglipTokenizer=siglipTokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)



def train(attn_implementation=None):
    global local_rank
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, VisionTowerArguments, VisionGeneratorArguments, ReconstructionLossArguments, ConstrastiveLossArguments))
    model_args, data_args, training_args, vision_tower_args, vision_generator_args, rec_loss_args, constrative_loss_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    print('model_args:', model_args)
    print('data_args: ', data_args)
    
    model = SeTok(tokenizer_config=VisionTowerArguments,
                  detokenizer_config=VisionGeneratorArguments,
                  rec_loss_config=rec_loss_args, 
                  contrastive_loss_config=constrative_loss_args)

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

    
    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args,
                                              siglipTokenizer=siglipTokenizer)
    trainer = SetokTrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    model.config.use_cache = True
    safe_save_model_for_hf_trainer(trainer=trainer,
                                    output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
