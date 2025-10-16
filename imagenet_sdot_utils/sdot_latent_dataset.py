import torch
from tqdm import tqdm
from torch.utils.data import Dataset
import numpy as np
import os

NUM_CLASSES_IMAGENET = int(os.environ["NUM_CLASSES_IMAGENET"])
TEMP_DIR = os.environ["TEMP_DIR"]


class SdotRebalanceDatasetSeed(Dataset):
    def __init__(self, total_samples_needed=None):
        super().__init__()
        self.total_samples_needed = total_samples_needed
        self._load()

    def __getitem__(self, idx):
        # do not use jax in torch dataset. Otherwise, num_workers in dataloader cannot >0
        key = self.keys[idx]
        idx_imagenet = self.target_idx[idx]
        vae_mean = self.vae_mean_tensor[idx_imagenet]
        vae_std = self.vae_std_tensor[idx_imagenet]
        label = self.label_tensor[idx_imagenet]
        image_latent = vae_mean + vae_std * np.random.normal(size=vae_mean.shape)
        key = torch.from_numpy(np.array(key))
        return image_latent, key, label

    def __len__(self):
        return self.total_length

    def _load(self):
        vae_mean_path = os.path.join(TEMP_DIR, f"vae_mean.npy")
        vae_std_path = os.path.join(TEMP_DIR, f"vae_std.npy")
        label_path = os.path.join(TEMP_DIR, f"label.npy")

        self.vae_mean_tensor = np.load(vae_mean_path)
        self.vae_std_tensor = np.load(vae_std_path)
        self.label_tensor = np.load(label_path)

        keys_list = [None] * NUM_CLASSES_IMAGENET
        target_idx_in_all_list = [None] * NUM_CLASSES_IMAGENET
        for class_id in tqdm(range(NUM_CLASSES_IMAGENET)):
            keys_path = os.path.join(TEMP_DIR, f"keys_{class_id}.npy")
            target_idx_in_all_path = os.path.join(
                TEMP_DIR, f"target_idx_in_all_{class_id}.npy"
            )

            keys = np.load(keys_path)
            target_idx_in_all = np.load(target_idx_in_all_path)

            keys_list[class_id] = keys
            target_idx_in_all_list[class_id] = target_idx_in_all

        self.keys = np.concatenate(keys_list)
        self.target_idx = np.concatenate(target_idx_in_all_list)
        # self.total_length=len(self.keys)
        self.total_length = len(self.keys)
        assert self.total_length == len(self.target_idx)
        print("total_samples_needed: " + str(self.total_samples_needed))
        print("self.total_length: " + str(self.total_length))
        if self.total_samples_needed != None:
            assert self.total_samples_needed < self.total_length


class TrainDatasetLatent(Dataset):
    def __init__(self):
        super().__init__()
        self._load()

    def __getitem__(self, idx):
        vae_mean = self.vae_mean_tensor[idx]
        vae_std = self.vae_std_tensor[idx]
        labels = self.label_tensor[idx]
        image_latent = vae_mean + vae_std * np.random.normal(size=vae_mean.shape)
        return image_latent, 0, labels

    def __len__(self):
        return len(self.vae_mean_tensor)

    def _load(self):
        vae_mean_path = os.path.join(TEMP_DIR, f"vae_mean.npy")
        vae_std_path = os.path.join(TEMP_DIR, f"vae_std.npy")
        label_path = os.path.join(TEMP_DIR, f"label.npy")

        self.vae_mean_tensor = np.load(vae_mean_path)
        self.vae_std_tensor = np.load(vae_std_path)
        self.label_tensor = np.load(label_path)


class ValidateDatasetLatent(Dataset):
    def __init__(self):
        super().__init__()
        self._load()

    def __getitem__(self, idx):
        vae_mean = self.vae_mean[idx]
        vae_std = self.vae_std[idx]
        label = self.labels[idx]
        image_latent = vae_mean + vae_std * np.random.normal(size=vae_mean.shape)
        # noise=np.asarray(noise)

        return image_latent, label

    def __len__(self):
        return len(self.vae_mean)
        # return 10000

    def _load(self):
        vae_mean_path = os.path.join(TEMP_DIR, f"vae_mean.npy")
        vae_std_path = os.path.join(TEMP_DIR, f"vae_std.npy")
        label_path = os.path.join(TEMP_DIR, f"label.npy")

        self.vae_mean = np.load(vae_mean_path)
        self.vae_std = np.load(vae_std_path)
        self.labels = np.load(label_path)
