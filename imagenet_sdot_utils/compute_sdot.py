import torch
import os
from sdot import sdot_solve
import numpy as np

TEMP_DIR = os.environ["TEMP_DIR"]

vae_mean_path = os.path.join(TEMP_DIR, f"vae_mean.npy")
label_path = os.path.join(TEMP_DIR, f"label.npy")
index_list_path = os.path.join(TEMP_DIR, f"index_list.npy")

vae_mean_tensor = np.load(vae_mean_path)
index_list = np.load(index_list_path, allow_pickle=True)


def compute_sdot_class(class_id, gpu_id):
    device = torch.device(f"cuda:{gpu_id}")
    dual_weight_path = os.path.join(TEMP_DIR, f"dual_weight_class_{class_id}.pt")
    if os.path.exists(dual_weight_path):
        return class_id, gpu_id
    data_tensor = vae_mean_tensor[index_list[class_id]]
    data_tensor = torch.from_numpy(data_tensor).to(device)
    data_tensor_flatten = data_tensor.reshape(data_tensor.shape[0], -1)
    num_targets = len(data_tensor)
    dual_weight = torch.zeros(num_targets, device=device)
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(dual_weight_path), exist_ok=True)
    pbar_args = {
        "desc": f"Class {class_id} on GPU {gpu_id}: {num_targets} targets",
        "position": gpu_id,
    }
    dual_weight, _ = sdot_solve(
        data_tensor_flatten,
        lr=10,
        num_step=3_000,
        batch_size=4096,
        dual_weight_init=dual_weight,
        eps_entropic=0.01,
        ema_param=0.99,
        pbar_args=pbar_args,
    )

    torch.save(dual_weight, dual_weight_path)
    return class_id, gpu_id


if __name__ == "__main__":
    compute_sdot_class(10, 0)
