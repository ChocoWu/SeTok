
import math
import random
from PIL import Image


def extend_list(original_list, multiplier):
    # Calculate how many elements to replicate and how many to select randomly
    replicate_elements = math.floor(multiplier)
    random_elements = multiplier - replicate_elements

    # Replicate the list
    replicated_list = original_list * replicate_elements

    # Calculate how many elements to randomly select
    select_elements = math.ceil(len(original_list) * random_elements)

    # Randomly select elements and append to the replicated list
    for _ in range(select_elements):
        random_element = random.choice(original_list)
        replicated_list.append(random_element)

    return replicated_list


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result