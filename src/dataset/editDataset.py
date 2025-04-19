import pdb
import token

from datasets import load_from_disk
import io
import numpy as np
from PIL import Image
import random
import torch
from torchvision import transforms
from torch.utils.data import Dataset, ConcatDataset
from .base_dataset import *
from .dataset_utils import expand2square


def convert_to_np(image, resolution):
    image = image.convert("RGB")
    image = image.resize((resolution, resolution), resample=Image.Resampling.BICUBIC)
    return np.array(image).transpose(2, 0, 1)


def load_img_for_generator(image, resolution):
    # image = Image.open(path).convert("RGB")
    # w, h = image.size
    # print(f"loaded input image of size ({w}, {h}) from {path}")
    # w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((resolution), resample=Image.Resampling.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image.transpose(2, 0, 1)
    image = torch.from_numpy(image)
    return 2.*image - 1.


def get_random_response():
    image_editing_responses = {
        "simple": [
            "Here you go.",
            "All set.",
            "Done.",
            "Here it is.",
            "Finished.",
            "Done. Let me know if it works!"
        ],
        "polite_professional": [
            "The image has been edited as requested. Please take a look.",
            "Here is the updated version of the image you asked for.",
            "Attached is the revised image. Let me know if everything looks good.",
            "I've completed the edits—feel free to review.",
            "Please find the edited image below. Let me know if you'd like any revisions."
        ],
        "casual_friendly": [
            "All done! Hope you like it.",
            "Tada 🎨 Let me know what you think!",
            "Voila! Here's your image.",
            "Here's the new version—check it out!",
            "Done and dusted 😎",
            "Boom! Updated and ready."
        ],
        "open_to_feedback": [
            "Let me know if you'd like to adjust anything else.",
            "Happy to make further edits if needed!",
            "If you need a different version, just say the word.",
            "Want to tweak anything? I've got you.",
            "Tell me if something needs changing!"
        ],
        "image_generation_context": [
            "Here is the image based on your description.",
            "The generated image is ready. Let me know if it matches your vision.",
            "Here's what I came up with—does this align with what you had in mind?",
            "Based on your prompt, this is the result. Happy to revise!"
        ]
    }
    
    all_responses = sum(image_editing_responses.values(), [])

    random_reply = random.choice(all_responses)
    return random_reply



class EditingDataset(Dataset):
    def __init__(self, data_path, tokenizer, data_args) -> None:
        super().__init__()

        instructPix2Pix_dataset = InstructPix2Pix_Dataset(data_path[0], tokenizer=tokenizer, data_args=data_args)
        magicBruch_dataset = MagicBrush_Dataset(data_path=data_path[1], tokenizer=tokenizer, data_args=data_args)
        self.datasets = ConcatDataset([instructPix2Pix_dataset, magicBruch_dataset])
    
    def __len__(self):
        return self.datasets.__len__()

    def __getitem__(self, item):
        return self.datasets.__getitem__(item)



# InstructPix2Pix dataset
class InstructPix2Pix_Dataset(LazySupervisedDataset):
    '''
    according to InstructPix2Pix, the dataset can be used to train models to follow edit instructions.
    Edit instructions are available in the 'edit_prompt'. 'original_image' can be used with the 'edit_prompt' and 'edited_image' denotes the image after applying the 'edit_prompt' on the 'original_image'.
    "original_image" + "edited_image" + "edit_prompt"
    '''
    def __init__(self,
                 data_path,
                 tokenizer,
                 data_args,
                 ):
        super().__init__(data_args=data_args, data_path=data_path, tokenizer=tokenizer)

        # InstructPix2Pix Dataset path
        self.list_data_dict = load_from_disk(data_path)
        # 224, 256
        self.resolution_for_comp = data_args.image_size
        self.resolution_for_gen = data_args.resolution_sd

        # tokenizer
        self.tokenizer = tokenizer


    def __len__(self,):
        return len(self.list_data_dict)

    def __getitem__(self, i):
        # # {'original_image': <PIL.Image.Image image mode=RGB size=512x512 at 0x7F3879D3E4C0>, 'edited_image': <PIL.Image.Image image mode=RGB size=512x512 at 0x7F3879D3E460>, 'edit_prompt': 'make the leaves yellow'}
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'image' in sources[0]:
            original_image_file = self.list_data_dict[i]['original_image']
            # image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            # image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            original_image = Image.open(io.BytesIO(original_image_file['bytes'])).convert('RGB')
            edited_image_file = self.list_data_dict[i]['edited_image']
            edited_image = Image.open(io.BytesIO(edited_image_file['bytes'])).convert('RGB')
            if self.data_args.image_aspect_ratio == 'pad':
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
                original_image = expand2square(original_image, tuple(int(x*255) for x in processor.image_mean))
                comp_image = processor.preprocess(original_image, return_tensors='pt')['pixel_values'][0]
                gen_image =  load_img_for_generator(edited_image, self.resolution_for_gen)
            else:
                comp_image = processor.preprocess(original_image, return_tensors='pt')['pixel_values'][0]
                gen_image = load_img_for_generator(edited_image, self.resolution_for_gen)
            
            _source = {
                "id": i,
                "conversations": [
                    {"from": "human", "value": "<image>\n"+self.list_data_dict[i]["edit_prompt"]},
                    {"from": "gpt", "value": "<Target>\n"+ get_random_response()},
                ]
            }
            sources = preprocess_multimodal(
                [_source],
                self.data_args,
                target_num=gen_image.shape[0])
        else:
            sources = [_source]
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]))
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['comp_image'] = comp_image
            data_dict['gen_image'] = gen_image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['comp_image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
            data_dict['gen_image'] = torch.zeros(3, crop_size['height'], crop_size['width'])

        return data_dict



# MagicBrush dataset
class MagicBrush_Dataset(LazySupervisedDataset):
    '''
    according to MagicBrush, the dataset can be used to train models to follow edit instructions.
    Edit instructions are available in the 'instruction'. 'source_img' can be used with the 'instruction' and 'target_img' denotes the image after applying the 'instruction' on the 'source_img'.
    "source_img" + "target_img" + "instruction"
    Dataset({features: ['img_id', 'turn_index', 'source_img', 'mask_img', 'instruction', 'target_img'], num_rows: 8807})
    '''
    def __init__(self,
                 data_path,
                 tokenizer,
                 data_args,
                 ):
        super().__init__(data_path=data_path, tokenizer=tokenizer, data_args=data_args)
        # MagicBrush Dataset path
        # InstructPix2Pix Dataset path
        self.list_data_dict = load_from_disk(data_path)
        # 224, 256
        self.resolution_for_comp = data_args.image_size
        self.resolution_for_gen = data_args.resolution_sd

        # tokenizer
        self.tokenizer = tokenizer

    def __len__(self,):
        return len(self.list_data_dict)

    def __getitem__(self, i):
        # {'source_img': <PIL.Image.Image image mode=RGB size=500x500 at 0x7F327BE01100>, 'target_img': <PIL.Image.Image image mode=RGB size=1024x1024 at 0x7F327BE010D0>, 'instruction': 'let the asparagus be replaced with sausages'}
        
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'image' in sources[0]:
            original_image_file = self.list_data_dict[i]['source_img']
            # image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            # image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            original_image = Image.open(io.BytesIO(original_image_file['bytes'])).convert('RGB')
            edited_image_file = self.list_data_dict[i]['target_img']
            edited_image = Image.open(io.BytesIO(edited_image_file['bytes'])).convert('RGB')
            if self.data_args.image_aspect_ratio == 'pad':
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
                original_image = expand2square(original_image, tuple(int(x*255) for x in processor.image_mean))
                comp_image = processor.preprocess(original_image, return_tensors='pt')['pixel_values'][0]
                gen_image =  load_img_for_generator(edited_image, self.resolution_for_gen)
            else:
                comp_image = processor.preprocess(original_image, return_tensors='pt')['pixel_values'][0]
                gen_image = load_img_for_generator(edited_image, self.resolution_for_gen)
            
            _source = {
                "id": i,
                "conversations": [
                    {"from": "human", "value": "<image>\n"+self.list_data_dict[i]["instruction"]},
                    {"from": "gpt", "value": "<image>\n"+ get_random_response()},
                ]
            }
            sources = preprocess_multimodal(
                [_source],
                self.data_args,
                target_num=gen_image.shape[0])
        else:
            sources = [_source]
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]))
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['comp_image'] = comp_image
            data_dict['gen_image'] = gen_image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['comp_image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
            data_dict['gen_image'] = torch.zeros(3, crop_size['height'], crop_size['width'])

        return data_dict