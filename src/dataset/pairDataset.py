import pdb

from datasets import load_from_disk
import io
import numpy as np
from PIL import Image
import random
import torch
from typing import Dict, Optional, Sequence, List, Union
from torchvision import transforms
from torch.utils.data import Dataset
from transformers.trainer_pt_utils import LabelSmoother

from src.mm_utils import get_anyres_image_grid_shape

from .base_dataset import LazySupervisedDataset
from .base_dataset import *
from .dataset_utils import extend_list, expand2square
import PIL
import os
import json
import transformers
from tqdm import tqdm
import time
from pycocotools.coco import COCO


def convert_to_np(image, resolution):
    image = image.convert("RGB")
    image = image.resize((resolution, resolution), resample=Image.Resampling.BICUBIC)
    return np.array(image).transpose(2, 0, 1)


# all the data will be preprocessed into a unified format, [{"caption":"xxx", "image": "image_apth"}, ...]
def load_cc3m(data_path=None, image_folder=None, is_only_load=False):

    # cc3m = 
    assert (data_path is not None) or (image_folder is not None), "data_path and image_folder should not be None at the same time."

    if data_path is None:
        data = []
        # Walk through the directory
        for filename in tqdm(os.listdir(image_folder)):
            # Check for files ending with .json
            if filename.endswith('.json'):
                file_path = os.path.join(image_folder, filename)
                try:
                    # Open and load the JSON file
                    with open(file_path, 'r') as file:
                        _data = json.load(file)
                        data.append(_data)
                    # print(f"Loaded {filename}")
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON from {filename}: {e}")
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
        # Write the merged JSON data to the output file
        try:
            data_path = os.path.join(os.path.dirname(image_folder), 'cc3m.json')
            with open(data_path, 'w') as outfile:
                json.dump(data, outfile, indent=4)
            print(f"Merged data written to {data_path}")
        except Exception as e:
            print(f"Error writing to {data_path}: {e}")
    else:
        with open(data_path, 'r') as file:
            data = json.load(file)
    
    final_data = []
    for d in data:
        _instance = {
            'caption': d['caption'], 
            'image': os.path.join(image_folder, d['key']+'.jpg')
            }
        _instance['dataset'] = 'cc3m'
        final_data.append(_instance)
    return final_data


def load_coco(dataset_name: str, data_path: List[str], image_folder: str, is_only_load: bool=True):
    coco_captions = COCO(os.path.join(data_path, 'captions_train2017.json'))
    img_ids = sorted(coco_captions.getImgIds())

    tic = time.time()
    panoticToAnn = {}
    with open(os.path.join(data_path, 'panoptic_train2017.json'), 'r') as f:
        panotic_dataset = json.load(f)
    if 'annotations' in panotic_dataset:
        for ann in panotic_dataset['annotations']:
            panoticToAnn[ann['file_name']] = ann
    print('Panotic Data Load Done (t={:0.2f}s)'.format(time.time()- tic))
    # curr_conv = uvlm_conv_v1
    cluster_num_list = []
    skip_idx = 0
    final_data = []

    categories = panotic_dataset['categories']
    category_id_to_name = {}
    for cate in categories:
        category_id_to_name[cate['id']] = cate['name']
    
    for img_id in tqdm(img_ids):
        ann_img = coco_captions.loadImgs(img_id)
        image_w = float(ann_img[0]["width"])
        image_h = float(ann_img[0]["height"])
        imageName = ann_img[0]["file_name"]

        # obtain pannotic annotations
        pannotic_name = imageName.split('.')[0]+'.png'
        panotic_anno = panoticToAnn[pannotic_name]
        segmentations = []
        phrases = []
        
        if len(panotic_anno['segments_info']) > 0:
            segmentations = panotic_anno['segments_info']
            phrases = [category_id_to_name[seg['category_id']] for seg in panotic_anno['segments_info']]
            # bboxes = [seg for seg in panotic_anno['segment_info']]
        else: 
            skip_idx += 1
            continue
        cluster_num_list.append(len(segmentations))
        if not is_only_load:
            # if os.path.exists(os.path.join(image_folder, imageName)):
            imageName = os.path.join(image_folder, imageName)
            # else:
            #     print(os.path.join(image_folder, imageName)) 
        elements_instance = coco_captions.loadAnns(coco_captions.getAnnIds(imgIds=[img_id]))
        if len(elements_instance) > 0:
            caption = elements_instance[0]['caption']
        else:
            continue

        _instance = {
            "id": imageName,
            "image": imageName,
            "cluster_num": min(len(segmentations), 74),
            "caption": caption,
            "phrases": ','.join(phrases),
        }
        _instance['dataset'] = 'coco2017'
        final_data.append(_instance)
    # Create the scatter plot
    x = range(len(cluster_num_list))
    # plt.scatter(x, cluster_num_list)

    # # Display the plot
    # plt.savefig('val_cluster_num.png')
    # plt.show()
    return final_data


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


def load_test():
    data = [
        {
            'caption': 'A brown horse standing in a field with a single large oak tree and a distant mountain.', 
            'image': '/mnt/shengqiongwu/uvlm/vlmYi/LLaVA_2/test_imgs/horse.png',
            'dataset': 'test'
        },
        {
            'caption': 'a view of ocean', 
            'image': '/mnt/shengqiongwu/uvlm/vlmYi/LLaVA_2/test_imgs/sea.png',
            'dataset': 'test'
        }
    ]
    return data



def get_random_generation_response():
    image_generation_responses = {
        "simple": [
            "Here you go.",
            "All set.",
            "Done.",
            "Here it is.",
            "Finished.",
            "Done. Let me know if it works!"
        ],
        "polite_professional": [
            "The image has been generated as requested. Please take a look.",
            "Here is the updated version of the image you asked for.",
            "Attached is the revised image. Let me know if everything looks good.",
            "I've completed the edits—feel free to review.",
            "Please find the generated image below. Let me know if you'd like any revisions."
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
    
    all_responses = sum(image_generation_responses.values(), [])

    random_reply = random.choice(all_responses)
    return random_reply

def get_random_captioning_instruction():
    basic_captioning_prompts = [
        "Describe the image.",
        "What is shown in the picture?",
        "Write a caption for this image.",
        "Summarize the content of this image.",
        "Give a one-sentence description of the image.",
        "What can be seen in this image?",
        "Write a natural description of this image.",
        "State what this image depicts.",
        "Caption the image in a neutral tone.",
        "Briefly describe the photo."
    ]
    concise_captioning_prompts = [
        "Generate a short caption (less than 10 words).",
        "Describe the image using a concise phrase.",
        "Give a short title-like description.",
        "Write a brief, tweet-length image caption.",
        "Use a few words to summarize the image."
    ]
    detailed_captioning_prompts = [
        "Provide a detailed description of the scene, objects, and people.",
        "Describe the image with full sentences and visual details.",
        "List all key visual elements present in the image.",
        "Describe the background, actions, and context of the scene.",
        "Write a structured paragraph that explains what's happening in the image."
    ]
    story_style_prompts = [
        "Tell a short story based on this image.",
        "Imagine a scene—what's the story behind this picture?",
        "Write a fictional scenario that explains this image.",
        "Describe this image as if it's a scene in a movie.",
        "Create a narrative based on the characters and setting in the image."
    ]
    social_media_prompts = [
        "Write a fun caption for this Instagram post.",
        "What's a witty description for this image?",
        "Caption this image like you're posting it on Twitter.",
        "Give this image a viral-worthy description.",
        "Write a casual, friendly caption for social sharing."
    ]
    vision_assist_prompts = [
        "Describe the image for someone who cannot see it.",
        "Provide a detailed and objective explanation of the scene.",
        "Write an accessible image description for screen readers.",
        "Explain clearly what's in the photo for a visually impaired user.",
        "Describe this image with visual details and context cues."
    ]
    all_captioning_prompts = (
        basic_captioning_prompts +
        concise_captioning_prompts +
        detailed_captioning_prompts +
        story_style_prompts +
        social_media_prompts +
        vision_assist_prompts
    )

    random_reply = random.choice(all_captioning_prompts)
    return random_reply



# Text-Image Pair dataset
class TextImagePairDataset(LazySupervisedDataset):
    '''
    Preparing the text_imagepair dataset tune the diffusion for image generator. we can use multiple dataset to train the model together. 
    '''
    def __init__(self,
                 data_path: Union[str, List[str]],
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args,
                 constrative_tokenizer: transformers.PreTrainedTokenizer=None,
                 vision_tokenizer: nn.Module=None,
                 ):
        """
        Args:
            data_path: Path to the dataset.
            image_root: Path to the image root.
            resolution_vit: Resolution for the ViT model.
            resolution_sd: Resolution for the Style Discriminator.
            CLIPImageProcessor: CLIPImageProcessor object.
            CLIPTokenizer: CLIPTokenizer object.
        """
        super().__init__(data_path=data_path, tokenizer=tokenizer, data_args=data_args)
        list_data_dict = []
        data_paths = data_path if isinstance(data_path, list) else [data_path]
        image_folders = data_args.image_folder if isinstance(data_args.image_folder, list) else [data_args.image_folder]
        dataset_names = data_args.dataset_name if isinstance(data_args.dataset_name, list) else [data_args.dataset_name]

        print(f"Loading data from {data_paths}")
        print(f"Loading images from {image_folders}")
        print(f"Loading dataset names {dataset_names}")
        # print(f"Loading data multiple {data_multiple}")

        for data_path_i, image_folder_i, dataset_name_i in zip(data_paths, image_folders, dataset_names):
            if dataset_name_i in 'cc3m':
                datas = load_cc3m(data_path_i, image_folder_i, is_only_load=False)
                list_data_dict.append(datas)
            elif dataset_name_i in 'coco2017':
                datas = load_coco(dataset_name_i, data_path_i, image_folder_i, is_only_load=False)
                list_data_dict.append(datas)          
            else:
                raise ValueError(f"Unknown dataset {data_path_i}")
        
        data_multiple = data_args.data_multiple
        if data_multiple is None:
            # Concat all data directly and Shuffle.
            list_data_dict = [item for dataset_i in list_data_dict for item in dataset_i]
            random.shuffle(list_data_dict)
        else:
            new_list_data_dict = []
            for data_scaler_i, dataset_i in zip(data_multiple, list_data_dict):
                dataset_name_i = dataset_i[0]['dataset']
                new_dataset_i = extend_list(dataset_i, data_scaler_i)
                new_list_data_dict.extend(new_dataset_i)
            list_data_dict = new_list_data_dict
            random.shuffle(list_data_dict)
        
        print('the number of data:', len(list_data_dict))
        self.list_data_dict = list_data_dict[:240000]
        
        # 224, 256
        self.resolution_ViT = data_args.image_size
        self.resolution_gen = data_args.resolution_gen

   
        # LLM tokenizer
        self.text_tokenizer = tokenizer
        self.text_tokenizer.padding_side = "right"
        self.text_tokenizer.truncation_side = 'right'

        # Contrasitive Tokenizer
        self.constrative_tokenizer = constrative_tokenizer
        self.constrative_tokenizer.padding_side = "right"
        self.constrative_tokenizer.truncation_side = 'right'

        self.vision_tokenizer = vision_tokenizer


    def __len__(self,):
        return len(self.list_data_dict)

    def __getitem__(self, i):
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]

        if self.data_args.task_type == 'caption':
            _source = {
                "id": i,
                "conversations": [
                    {"from": "human", "value": "<image>\n"+ get_random_captioning_instruction()},
                    {"from": "gpt", "value": self.list_data_dict[i]["caption"]},
                ]
            }
        elif self.data_args.task_type == 'generation':
            _source = {
                "id": i,
                "conversations": [
                    {"from": "human", "value": "<image>\n"+self.list_data_dict[i]["caption"]},
                    {"from": "gpt", "value": "<Target>\n"+ get_random_generation_response()},
                ]
            }
        else:
            _source = {
                "id": i,
                "conversations": [
                    {"from": "human", "value": self.list_data_dict[i]["question"]},
                    {"from": "gpt", "value": self.list_data_dict[i]["answer"]},
                ]
            }
    
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            # image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            # image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            image = Image.open(image_file).convert('RGB')
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
                image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                comp_image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                gen_image = self.vision_tokenizer(image)
            else:
                gen_image = self.vision_tokenizer(image)
                comp_image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            sources = preprocess_multimodal(
                [_source],
                self.data_args,
                target_num=gen_image.shape[0]
            )
        else:
            sources = [_source]
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]))
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['comp_image'] = comp_image
            data_dict['gen_image'] = gen_image
            data_dict['num_tokens'] = gen_image.shape[0]
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['comp_image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
            data_dict['gen_image'] = torch.zeros(64, 4096)
        
        if 'phrases' in self.list_data_dict[i]:
            input_ids_for_constrative = self.constrative_tokenizer(
                self.list_data_dict[i]['phrases'],
                return_tensors="pt",
                padding="max_length",
                max_length=self.constrative_tokenizer.model_max_length,
                truncation=True,
            ).input_ids[0]
        else:   
            input_ids_for_constrative = self.constrative_tokenizer(
                self.list_data_dict[i]['caption'],
                return_tensors="pt",
                padding="longest",
                max_length=self.constrative_tokenizer.model_max_length,
                truncation=True,
            ).input_ids[0]
        data_dict.update({
            'input_ids_for_constrative': input_ids_for_constrative,
            'caption': self.list_data_dict[i]['caption'],
            'phrases': self.list_data_dict[i]['phrases'] if 'phrases' in self.list_data_dict[i] else self.list_data_dict[i]['caption'],
        })

        return data_dict
    
