from dataclasses import asdict, is_dataclass
from ..setok import SetokDeTokenizer

def build_vision_generator(image_generator_cfg, **kwargs):
    if is_dataclass(image_generator_cfg):
        image_generator_cfg = asdict(image_generator_cfg)
    elif isinstance(image_generator_cfg, dict):
        image_generator_cfg = image_generator_cfg
    else:
        image_generator_cfg = vars(image_generator_cfg)

    return SetokDeTokenizer(**image_generator_cfg, **kwargs)