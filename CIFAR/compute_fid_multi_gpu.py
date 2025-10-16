# modified from torchcfm: https://github.com/atong01/conditional-flow-matching/blob/main/examples/images/cifar10/compute_fid.py
# multi gpu support is added.
# Usage: TODO

import os
import sys

import torch
from cleanfid import fid
from torchdiffeq import odeint
from torchdyn.core import NeuralODE

from torchcfm.models.unet.unet import UNetModelWrapper
import torch.distributed as dist
import pandas as pd
from glob import glob

# Settings
batch_size_fid=1024
integration_method='euler' # euler or dopri5
euler_steps=1000 # only used when integration_method='euler'
path_pattern='./results/*/*400000.pt'
# result_csv_path='./fid_result_dopri5.csv'
result_csv_path='./fid_result_1000.csv'


path_list=glob(path_pattern)

def setup_distributed():
    """Set up distributed environment"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
        return True, rank, world_size, local_rank
    return False, 0, 1, 0

# Set up distributed environment
is_distributed, rank, world_size, local_rank = setup_distributed()
print(is_distributed)
local_device = f"cuda:{local_rank}" if is_distributed else "cuda:0" if torch.cuda.is_available() else "cpu"
fid_device_id=0
# Define the model
new_net = UNetModelWrapper(
    dim=(3, 32, 32),
    num_res_blocks=2,
    num_channels=128,
    channel_mult=[1, 2, 2, 2],
    num_heads=4,
    num_head_channels=64,
    attention_resolutions="16",
    dropout=0.1,
).to(local_device)

# Load the model
if not os.path.exists(result_csv_path):
    column_names = ['path', 'fid-50k']
    df = pd.DataFrame(columns=column_names)
    df.to_csv(result_csv_path, index=False)
df = pd.read_csv(result_csv_path, header=0)
path_list = [p for p in path_list if p not in list(df['path'])]
if len(path_list)==0:
    exit(1)
path=path_list[0]
if rank == 0:
    print("path: ", path)

checkpoint = torch.load(path, map_location=local_device)
state_dict = checkpoint["ema_model"]

try:
    new_net.load_state_dict(state_dict)
except RuntimeError:
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[k[7:]] = v
    new_net.load_state_dict(new_state_dict)
new_net.eval()


if rank == 0:
    print("Start computing FID")


if integration_method == "euler":
    node = NeuralODE(new_net, solver=integration_method).to(local_device)


def gen_1_img(unused_latent):
    """Generator function that works with distributed setup"""
    local_batch_size = batch_size_fid // world_size if is_distributed else batch_size_fid
    
    with torch.no_grad():
        # Generate batches on the current GPU
        x = torch.randn(local_batch_size, 3, 32, 32, device=local_device)
        
        if integration_method == "euler":
            print("Use method: ", integration_method)
            t_span = torch.linspace(0, 1, euler_steps + 1, device=local_device)
            traj = node.trajectory(x, t_span=t_span)
        else:
            t_span = torch.linspace(0, 1, 2, device=local_device)
            traj = odeint(
                new_net, x, t_span, rtol=1e-5, atol=1e-5, 
                method=integration_method
            )
    
    traj = traj[-1, :]
    local_img = (traj * 127.5 + 128).clip(0, 255).to(torch.uint8)

        # If not distributed, just return the local image
    if not is_distributed:
        return local_img
    
    # Otherwise, we need to gather all images from all processes
    # First, get the shape of the tensor to gather
    img_gathered = [torch.zeros_like(local_img) if i!=local_rank else local_img for i in range(world_size) ]    
    # Create a list to hold tensors from all processes
    
    # All gather actually happens here
    dist.barrier() 
    for i in range(world_size):
        dist.broadcast(img_gathered[i], src=i)
    
    # Concatenate all images along batch dimension
    gathered_img = torch.cat(img_gathered, dim=0)
    
    return gathered_img
def compute_distributed_fid():
    # Use CleanFID with the distributed option
    try:
        score = fid.compute_fid(
            gen=gen_1_img,
            dataset_name="cifar10",
            batch_size=batch_size_fid,
            dataset_res=32,
            num_gen=50000,
            dataset_split="train",
            mode="legacy_tensorflow",
            device=torch.device("cuda:"+str(fid_device_id)),
        )
        
        if not is_distributed or rank == 0:
            print(f"Final FID: {score}")
            return score
    except Exception as e:
        if not is_distributed or rank == 0:
            print(f"Error computing FID: {e}")
        raise
    
    return score

# Call this function to compute FID
if __name__ == "__main__":
    
    if not os.path.exists(result_csv_path):
        column_names = ['path', 'fid-50k']
        df = pd.DataFrame(columns=column_names)
        df.to_csv(result_csv_path, index=False)
    
    fid_score=compute_distributed_fid()
    if rank == 0:
        print('-------')
        print('path:')
        print(path)
        print('fid:')
        print(fid_score)
        new_row_data = {'path': path, 'fid-50k': fid_score}
        new_row_df = pd.DataFrame([new_row_data])
        new_row_df.to_csv(result_csv_path, mode='a', header=False, index=False)
    exit(0)