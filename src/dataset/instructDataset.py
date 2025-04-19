
import pdb
import os
import torch
import transformers
import random
from tqdm import tqdm
from typing import List, Optional, Union
from torch.utils.data import Dataset
from PIL import Image
from .dataset_utils import extend_list
from .base_dataset import *
import json
from .vqa import VQA


def load_dataset(dataset_name: str, data_path: str, image_folder: str, is_only_load: bool=True):
    datas = json.load(open(data_path, "r"))
    for data_i in datas:
        data_i["dataset"] = dataset_name
        if not is_only_load:
            data_i["image"] = os.path.join(image_folder, data_i['image'])
    return datas


def load_llava_150k(dataset_name: str, data_path: str, image_folder: str, is_only_load: bool=True):
    datas = json.load(open(data_path, "r"))
    for data_i in datas:
        data_i["dataset"] = dataset_name
        # data_i["conversations"] = data_i["conversation"]
        # del data_i["conversation"]
        if not is_only_load:
            data_i["image"] = os.path.join(image_folder, data_i['image'])
    return datas


def load_GQA(data_path: str, image_folder: str, is_only_load: bool=True):
    if not isinstance(data_path, list):
        data_path = [data_path]
    
    final_data = []
    for data_path_i in data_path:
        datas = json.load(open(data_path_i, "r"))
        for k, v in tqdm(datas.items()):
            imageId = v['imageId']
            question = v['question']
            answer = v['fullAnswer']
            imageName = f'{imageId}.jpg'
            if not is_only_load:
                # if os.path.exists(os.path.join(image_folder, imageName)):
                imageName = os.path.join(image_folder, imageName)
                # else:
                #     print(os.path.join(image_folder, imageName))
            _instance = {
                "id": imageName,
                "image": imageName,
            }
            conversations = []
            conversations.append({
                'from': 'human', 
                'value': f'<image>\n{question} Please provide an accurate answer consisting of only one word or phrase.'
                })
            conversations.append({
                'from': 'gpt', 
                'value': answer
                })
            _instance['conversations'] = conversations
            _instance['dataset'] = 'gqa'
            final_data.append(_instance)
    return final_data   


def load_VQAv2(data_path: str, image_folder: str, is_only_load: bool=True):
    # from vqa import VQA
    # set up file names and paths
    versionType ='v2_' # this should be '' when using VQA v2.0 dataset
    taskType    ='OpenEnded' # 'OpenEnded' only for v2.0. 'OpenEnded' or 'MultipleChoice' for v1.0
    dataType    ='mscoco'  # 'mscoco' only for v1.0. 'mscoco' for real and 'abstract_v002' for abstract for v1.0. 
    dataSubType ='train2014'
    annFile     ='%s/%s%s_%s_annotations.json'%(data_path, versionType, dataType, dataSubType)
    quesFile    ='%s/%s%s_%s_%s_questions.json'%(data_path, versionType, taskType, dataType, dataSubType)
    # imgDir      ='%s/Images/%s/%s/' %(data_path, dataType, dataSubType)
    # create vqa object and vqaRes object
    vqa = VQA(annFile, quesFile)

    final_data = []
    for quesId in tqdm(vqa.getQuesIds()):
        anns = vqa.loadQA(quesId)
        if len(anns) != 0:
            for ann in anns:
                quesId = ann['question_id']
                question = vqa.qqa[quesId]['question']
                imageId = ann['image_id']
                imageName = 'COCO_train2014_{:012d}.jpg'.format(imageId)
                if not is_only_load:
                    # if os.path.exists(os.path.join(image_folder, imageName)):
                    imageName = os.path.join(image_folder, imageName)
                    # else:
                    #     print(os.path.join(image_folder, imageName))
                _instance = {
                    "id": imageName,
                    "image": imageName,
                }
                conversations = []
                conversations.append({
                    'from': 'human', 
                    'value': f'<image>\n{question} Please provide an accurate answer consisting of only one word or phrase.'
                    })
                answer = ann['answers'][0]['answer']
                conversations.append({
                    'from': 'gpt', 
                    'value': answer
                    })
                _instance['conversations'] = conversations
                _instance['dataset'] = 'VQAv2'
                final_data.append(_instance)
    return final_data


def load_TextQA(data_path: str, image_folder: str, is_only_load: bool=True):
    datas = json.load(open(data_path, "r"))
    final_data = []
    for data_i in tqdm(datas['data']):
        question = data_i['question']
        imageId = data_i['image_id']
        imageName = f'{imageId}.jpg'
        if not is_only_load:
            if os.path.exists(os.path.join(image_folder, imageName)):
                imageName = os.path.join(image_folder, imageName)
            else:
                print(os.path.join(image_folder, imageName))
        _instance = {
            "id": imageName,
            "image": imageName,
        }
        conversations = []
        conversations.append({
            'from': 'human', 
            'value': f'<image>\n{question} Please provide an accurate answer.'
            })
        answer = data_i['answers'][0]
        conversations.append({
            'from': 'gpt', 
            'value': answer
            })
        _instance['conversations'] = conversations
        _instance['dataset'] = 'TextQA'
        final_data.append(_instance)
    return final_data


def load_AOKVQA(data_path: str, image_folder: str, is_only_load: bool=True):
    datas = json.load(open(data_path, "r"))
    final_data = []
    for data_i in datas:
        imageId = data_i['image_id']
        imageName = 'COCO_{}2014_{:012d}.jpg'.format(data_i['split'], imageId)
        if not is_only_load:
            # if os.path.exists(os.path.join(image_folder, imageName)):
            imageName = os.path.join(image_folder, imageName)
            # else:
            #     print(os.path.join(image_folder, imageName))
        _instance = {
            "id": imageName,
            "image": imageName,
        }
        conversations = []
        question = data_i['question']
        conversations.append({
            'from': 'human', 
            'value': f'<image>\n{question}'
            })
        answer = data_i['direct_answers'][0]
        rationale = data_i['rationales'][0]
        conversations.append({
            'from': 'gpt', 
            'value': f'{answer}. This is because {rationale}'
            })
        _instance['conversations'] = conversations
        _instance['dataset'] = 'AOKVQA'
        final_data.append(_instance)
    return final_data


def load_OKVQA(data_path: str, image_folder: str, is_only_load: bool=True):
    # from vqa import VQA
    # set up file names and paths
    versionType ='v2_' # this should be '' when using VQA v2.0 dataset
    taskType    ='OpenEnded' # 'OpenEnded' only for v2.0. 'OpenEnded' or 'MultipleChoice' for v1.0
    dataType    ='mscoco'  # 'mscoco' only for v1.0. 'mscoco' for real and 'abstract_v002' for abstract for v1.0. 
    dataSubType ='train2014'
    annFile     ='%s/%s_%s_annotations.json'%(data_path, dataType, dataSubType)
    quesFile    ='%s/%s_%s_%s_questions.json'%(data_path, taskType, dataType, dataSubType)
    imgDir      ='%s/Images/%s/%s/' %(data_path, dataType, dataSubType)
    # create vqa object and vqaRes object
    vqa = VQA(annFile, quesFile)

    final_data = []
    for quesId in tqdm(vqa.getQuesIds()):
        anns = vqa.loadQA(quesId)
        if len(anns) != 0:
            for ann in anns:
                quesId = ann['question_id']
                question = vqa.qqa[quesId]['question']
                imageId = ann['image_id']
                imageName = 'COCO_train2014_{:012d}.jpg'.format(imageId)
                if not is_only_load:
                    # if os.path.exists(os.path.join(image_folder, imageName)):
                    imageName = os.path.join(image_folder, imageName)
                    # else:
                    #     print(os.path.join(image_folder, imageName))
                _instance = {
                    "id": imageName,
                    "image": imageName,
                }
                conversations = []
                conversations.append({
                    'from': 'human', 
                    'value': f'<image>\n{question} Please provide an accurate answer consisting of only one word or phrase.'
                    })
                answer = ann['answers'][0]['answer']
                conversations.append({
                    'from': 'gpt', 
                    'value': answer
                    })
                _instance['conversations'] = conversations
                _instance['dataset'] = 'OKVQA'
                final_data.append(_instance)
    return final_data


# dataset for instruction tuning
class InstructionTuningDataset(LazySupervisedDataset):
    """ LLAVA-Dataset for instruction tuning """
    def __init__(self,
                 data_path: Union[str, List[str]],
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args,
                 ):
        super().__init__(data_path=data_path, tokenizer=tokenizer, data_args=data_args)

        data_paths = data_path if isinstance(data_path, list) else [data_path]
        image_folders = data_args.image_folder if isinstance(data_args.image_folder, list) else [data_args.image_folder]
        dataset_names = data_args.dataset_name if isinstance(data_args.dataset_name, list) else [data_args.dataset_name]

        print(f"Loading data from {data_paths}")
        print(f"Loading images from {image_folders}")
        print(f"Loading dataset names {dataset_names}")
        
        # ================================================
        list_data_dict = []
        for data_path_i, image_folder_i, dataset_name_i in zip(data_paths, image_folders, dataset_names):
            print(f"Loading {dataset_name_i} dataset")
            if 'LLaVA-CC3M-Pretrain-595K' in dataset_name_i:
                datas = load_dataset(dataset_name_i, data_path_i, image_folder_i, is_only_load=False)
                list_data_dict.append(datas)
            elif 'LLaVA150K' in dataset_name_i:
                datas = load_llava_150k(dataset_name_i, data_path_i, image_folder_i, is_only_load=False)
                list_data_dict.append(datas)
            elif 'LLaVA-LION-Pretrain' in dataset_name_i:
                datas = load_dataset(dataset_name_i, data_path_i, image_folder_i, is_only_load=False)
                list_data_dict.append(datas)
            elif 'ALLaVA-Caption-LAION-4V' in dataset_name_i:
                datas = load_dataset(dataset_name_i, data_path_i, image_folder_i, is_only_load=False)
                list_data_dict.append(datas)
            elif 'ALLaVA-Instruct-LAION-4V' in dataset_name_i:
                datas = load_dataset(dataset_name_i, data_path_i, image_folder_i, is_only_load=False)
                list_data_dict.append(datas)
            elif 'ShareGPT4V' in dataset_name_i:
                datas = load_dataset(dataset_name_i, data_path_i, image_folder_i, is_only_load=False)
                list_data_dict.append(datas)
            elif 'VQAv2' in dataset_name_i:
                datas = load_VQAv2(data_path_i, image_folder_i, is_only_load=False)
                list_data_dict.append(datas)
            elif 'OKVQA' == dataset_name_i:
                datas = load_OKVQA(data_path_i, image_folder_i, is_only_load=False)
                list_data_dict.append(datas)
            elif 'AOKVQA' == dataset_name_i:
                datas = load_AOKVQA(data_path_i, image_folder_i, is_only_load=False)
                list_data_dict.append(datas)
            elif 'GQA' in dataset_name_i:
                datas = load_GQA(data_path_i, image_folder_i, is_only_load=False)
                list_data_dict.append(datas)
            elif 'TextQA' in dataset_name_i:
                datas = load_TextQA(data_path_i, image_folder_i, is_only_load=False)
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
                print(f"Multiplying {dataset_name_i} by {data_scaler_i} times")
                new_dataset_i = extend_list(dataset_i, data_scaler_i)
                new_list_data_dict.extend(new_dataset_i)
            list_data_dict = new_list_data_dict
            random.shuffle(list_data_dict)

        print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        # self.list_data_dict = random.sample(list_data_dict, 400000)
        self.list_data_dict = list_data_dict
        self.data_args = data_args
    

    def __len__(self):
        return len(self.list_data_dict)


if __name__ == "__main__":

    # Test InstructionTuningDataset
    pass
    data_path = "./data/okvqa"
    image_folder = "./data/okvqa/images"
    # load_OKVQA(data_path, image_folder, is_only_load=True)

    data_path = "./data/vqa2"
    image_folder = "./data/okvqa/train2014"
    # load_VQAv2(data_path, image_folder, is_only_load=False)

    # data_path = "./data/gqa/train_all_questions/train_all_questions_0.json" # 1430536
    data_path = './data/gqa/train_balanced_questions.json'  # 943000
    image_folder = "./data/gqa/images"
    # data_path = [os.path.join(data_path, "train_balanced_questions.json"), os.path.join(data_path, "val_balanced_questions.json")]
    load_GQA(data_path, image_folder, is_only_load=False)


