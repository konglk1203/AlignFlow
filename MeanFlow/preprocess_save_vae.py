from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
import os
import numpy as np
import json
import sys

from dataset import LMDBLatentsDataset
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

data_dir="/home/ubuntu/code/MeanFlow/data/ImageNet/train_vae_latents_lmdb"
TEMP_DIR=os.environ['TEMP_DIR']
NUM_CLASSES_IMAGENET=int(os.environ['NUM_CLASSES_IMAGENET'])


latents_scale = torch.tensor(
        [0.18125, 0.18125, 0.18125, 0.18125]
        ).view(1, 4, 1, 1)
latents_bias = torch.tensor(
    [0., 0., 0., 0.]
    ).view(1, 4, 1, 1)


def save_vae():
    vae_mean_path=os.path.join(TEMP_DIR, f'vae_mean.npy')
    vae_std_path=os.path.join(TEMP_DIR, f'vae_std.npy')
    label_path=os.path.join(TEMP_DIR, f'label.npy')
    if os.path.exists(vae_mean_path) and os.path.exists(vae_std_path) and os.path.exists(label_path):
        return 
    os.makedirs(os.path.dirname(vae_mean_path), exist_ok=True)

    with open('./data/ImageNet/imagenet_class_to_idx.json', 'r') as f:
        class_to_idx = json.load(f)

    vae_mean_list=[]
    vae_std_list=[]
    label_list=[]
    flip_list=[False, True]
    for flip in flip_list:
        train_dataset = LMDBLatentsDataset(data_dir, flip_prob=1 if flip else 0)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=256,
            num_workers=0,
        )
        for moments, labels in tqdm(train_dataloader):
            posterior = DiagonalGaussianDistribution(moments)
            vae_mean, vae_std = posterior.mean, posterior.std
            vae_mean = vae_mean * latents_scale + latents_bias
            vae_std = vae_std * latents_scale
            vae_mean_list+=[vae_mean]
            vae_std_list+=[vae_std]

            labels = torch.tensor([class_to_idx[f'n{val:08d}'] for val in labels])
            label_list+=[labels]

            
    vae_mean_tensor=np.concatenate(vae_mean_list)
    vae_std_tensor=np.concatenate(vae_std_list)
    label_tensor=np.concatenate(label_list)

    np.save(vae_mean_path, vae_mean_tensor)
    np.save(vae_std_path, vae_std_tensor)
    np.save(label_path, label_tensor)


def generate_index_list():
    index_list_path=os.path.join(TEMP_DIR, f'index_list.npy')

    if os.path.exists(index_list_path):
        raise Exception('Already exists!')
    label_path=os.path.join(TEMP_DIR, f'label.npy')
    label_list=np.load(label_path)
    index_list = [[] for _ in range(NUM_CLASSES_IMAGENET)] 

    for i, label in enumerate(tqdm(label_list)):
        index_list[label]+=[i]
    np.save(index_list_path, np.array(index_list, dtype=object), allow_pickle=True)
if __name__ == "__main__":
    save_vae()
    generate_index_list()
