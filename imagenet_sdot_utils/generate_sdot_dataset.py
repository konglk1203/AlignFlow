import torch
from sdot import generate_sdot_dataset
import os
import numpy as np


MAX_NUM_EPOCH = int(os.environ["MAX_NUM_EPOCH"])
TEMP_DIR = os.environ["TEMP_DIR"]
BATCH_SIZE = 1024


vae_mean_path = os.path.join(TEMP_DIR, f"vae_mean.npy")
label_path = os.path.join(TEMP_DIR, f"label.npy")
index_list_path = os.path.join(TEMP_DIR, f"index_list.npy")

vae_mean_tensor = np.load(vae_mean_path)
index_list = np.load(index_list_path, allow_pickle=True)


def generate_sdot_dataset_class(class_id, gpu_id):
    device = f"cuda:{gpu_id}"
    device = torch.device(device)

    keys_path = os.path.join(TEMP_DIR, f"keys_{class_id}.npy")
    target_idx_in_all_path = os.path.join(TEMP_DIR, f"target_idx_in_all_{class_id}.npy")
    dual_weight_path = os.path.join(TEMP_DIR, f"dual_weight_class_{class_id}.pt")
    # if you do not want to recompute the dual weight, uncomment the following lines
    #
    # if os.path.exists(keys_path):
    #     return class_id, gpu_id
    os.makedirs(os.path.dirname(keys_path), exist_ok=True)

    data_tensor = vae_mean_tensor[index_list[class_id]]
    data_tensor = torch.from_numpy(data_tensor).to(device)
    dual_weight = torch.load(dual_weight_path, map_location=device)

    keys, target_idx_in_class = generate_sdot_dataset(
        data_tensor, dual_weight, batch_size=BATCH_SIZE, num_epoch=MAX_NUM_EPOCH
    )
    target_idx_in_all = torch.tensor(index_list[class_id], dtype=torch.int64)[
        target_idx_in_class
    ]
    target_idx_in_all = target_idx_in_all.numpy()

    np.save(keys_path, keys)
    np.save(target_idx_in_all_path, target_idx_in_all)
    return class_id, gpu_id


if __name__ == "__main__":
    generate_sdot_dataset_class(10, 0)
