from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List, Union
import transformers


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    vision_tokenizer: Optional[str] = field(default='setok')
    vision_generator: Optional[str] = field(default='setok')
    diff_loss: Optional[str] = field(default='diff_loss')
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    mm_use_im_start_end: bool = field(default=True)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default='flat')


@dataclass
class VisionTowerArguments:
    vision_tower: Optional[str] = field(default='google/siglip-so400m-patch14-384')
    pretrain_vision_tokenizer: Optional[str] = field(default='')
    unfreeze_mm_vision_tower: bool = field(default=False)
    mm_vision_select_feature: Optional[str] = field(default="patch")
    mm_vision_select_layer: Optional[int] = field(default=-1)
    delay_load: bool = field(default=False)
    hidden_dim: Optional[int] = field(default=4096)
    token_feat_dim: Optional[int] = field(default=4096)
    min_cluster_num: Optional[int] = field(default=64)
    threshold: Optional[float] = field(default=0.55)
    nheads: Optional[int] = field(default=2)
    dim_feedforward: Optional[int] = field(default=4096)
    proj_drop: Optional[float] = field(default=0.2)
    attn_drop: Optional[float] = field(default=0.0)
    inner_cluster_layers: Optional[int] = field(default=2)
    intra_cluster_layers: Optional[int] = field(default=2)

@dataclass
class VisionInProjectionArguments:
    pretrain_mm_in_mlp_adapter: Optional[str] = field(default=None)
    mm_in_projector_type: Optional[str] = field(default='mlp')
    mm_hidden_size: Optional[int] = field(default=1052)
    hidden_size: Optional[int] = field(default=4096)

@dataclass
class VisionGeneratorArguments:
    pretrain_vision_detokenizer: Optional[str] = field(default='')
    patch_size: Optional[int] = field(default=14)
    out_image_size: Optional[int] = field(default=384)
    decoder_embed_dim: Optional[int] = field(default=4096)
    decoder_nheads: Optional[int] = field(default=8)
    decoder_depth: Optional[int] = field(default=16)
    mlp_ratio: Optional[int] = field(default=4.0)
    feature_mapper_path_or_name: Optional[str] = field(default="bert-base-uncased")
    num_hidden_layers: Optional[int] = field(default=6)
    cross_attention_freq: Optional[int] = field(default=2)
    initializer_range: Optional[float] = field(default=0.02)

@dataclass 
class VisionOutProjectionArguments:
    pretrain_mm_out_mlp_adapter: Optional[str] = field(default=None)
    mm_out_projector_type: Optional[str] = field(default='mlp')


@dataclass
class ReconstructionLossArguments:
    disc_in_channels: Optional[int] = field(default=16)
    disc_num_layers: Optional[int] = field(default=2)
    disc_start: Optional[int] = field(default=5000)
    warm_up_end: Optional[int] = field(default=200)

@dataclass
class ConstrastiveLossArguments:
    text_encoder: Optional[str] = field(default='google/siglip-so400m-patch14-384')
    contrast_temperature: Optional[float] = field(default=0.07)
    multi_label: Optional[int] = field(default=0)
    share_temperature: bool = field(default=False)
    multi_label_loss_weight: Optional[float] = field(default=1.0)

@dataclass
class DiffLossArguments:
    diffloss_w: Optional[int] = field(default=3)
    diffloss_d: Optional[int] = field(default=1024)
    num_sampling_steps: Optional[str] = field(default='100')
    grad_checkpointing: bool = field(default=False)
    diffusion_batch_mul: Optional[int] = field(default=4)
    mask_ratio_min: Optional[float] = field(default=0.7)
    

@dataclass
class DataArguments:
    data_path: Union[List[str], str] = field(default=None, metadata={"help": "Path to the training data."})
    dataset_name: Union[List[str], str] = field(default=None)
    data_multiple: Union[List[float], str] = field(default=None, metadata={"help": "Data mutliplier for each dataset when mixed. None means direct concat."})  
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Union[List[str], str] = field(default=None)
    image_size: Optional[int] = field(default=448)
    image_aspect_ratio: str = 'square'
    task_type: Optional[str] = 'instruction'


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    tune_mm_in_mlp_adapter: bool = field(default=False)
    tune_mm_out_mlp_adapter: bool = field(default=False) 
    freeze_mm_in_mlp_adapter: bool = field(default=False)
    freeze_mm_out_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_in_projector_lr: Optional[float] = None
    mm_out_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)

    
