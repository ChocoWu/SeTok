import os
import json
from concurrent.futures import ThreadPoolExecutor
from joblib import Parallel, delayed

from tqdm import tqdm


def check_image_exists(data, image_folder):
    """ Helper function to check if an image exists and return the data accordingly. """
    if os.path.exists(os.path.join(image_folder, data['image'])):
        return (True, data)
    else:
        return (False, data['image'])


def preprocess(params):
    data_path, image_folder, save_path = params
    
    # Load data from JSON file
    with open(data_path, 'r') as f:
        datas = json.load(f)

    # Create a thread pool executor to check image existence in parallel
    new_datas = []
    unexisted_images = []
    with Parallel(n_jobs=50) as parallel:
        results = parallel(delayed(check_image_exists)(data, image_folder) for data in tqdm(datas, desc="Checking images"))


    # Separate the results into new data and unexisted images
    for result in results:
        exists, data = result
        if exists:
            new_datas.append(data)
        else:
            unexisted_images.append(data)
    
    # Save the filtered data back to a JSON file
    with open(save_path, 'w') as f:
        json.dump(new_datas, f, indent=4)
    
    # Print out the unexisted images
    print(f'Unexisted images: {unexisted_images}')



if __name__ == "__main__":
    data_path = './ALLaVA-4V/allava_laion/ALLaVA-Instruct-LAION-4V.json'
    image_folder = './ALLaVA-4V'
    save_path = './ALLaVA-4V/allava_laion/ALLaVA-Instruct-LAION-4V_preprocessed.json'

    import torch
    pretrain_mm_mlp_adapter = './checkpoints/vicuna-v1.5-7b-convnext-pretrain/mm_projector.bin'
    mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
    def get_w(weights, keyword):
        return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

    res = get_w(mm_projector_weights, 'mm_projector')