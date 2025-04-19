import os
from dataclasses import asdict, is_dataclass
from .clip_encoder import CLIPVisionTower
from ..setok.tokenizer import SetokTokenizer

def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'vision_tokenizer', getattr(vision_tower_cfg, 'vision_tower', None))
    # is_absolute_path_exists = os.path.exists(vision_tower)
    # if is_absolute_path_exists and (vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower or "vit" in vision_tower):
    #     return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    if is_dataclass(vision_tower_cfg):
        vision_tower_cfg = asdict(vision_tower_cfg)
    elif isinstance(vision_tower_cfg, dict):
        vision_tower_cfg = vision_tower_cfg
    else:
        vision_tower_cfg = vars(vision_tower_cfg)


    if 'siglip' in vision_tower:
        return SetokTokenizer(**vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')
