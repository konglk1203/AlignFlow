# Inspired from https://github.com/w86763777/pytorch-ddpm/tree/master.

# Authors: Kilian Fatras
#          Alexander Tong


import matplotlib.pyplot as plt
import torch
from absl import app, flags
from cleanfid import fid
from torchdiffeq import odeint
from torchdyn.core import NeuralODE

from torchcfm.models.unet.unet import UNetModelWrapper


integration_steps=100
integration_method='dopri5'
num_gen=10000
tol=1e-5
batch_size_fid=1024

def gen_1_img(unused_latent, net, device):
    with torch.no_grad():
        x = torch.randn(batch_size_fid, 3, 32, 32, device=device)
        t_span = torch.linspace(0, 1, 2, device=device)
        traj = odeint(
            net, x, t_span, rtol=tol, atol=tol, method=integration_method
        )
    traj = traj[-1, :]  # .view([-1, 3, 32, 32]).clip(-1, 1)
    img = (traj * 127.5 + 128).clip(0, 255).to(torch.uint8)  # .permute(1, 2, 0)
    return img


def get_fid(net, dataset_name="cifar10", num_gen=10000, device='cuda'):
    score=fid.compute_fid(gen=lambda unused_latent: gen_1_img(unused_latent, net, device), dataset_name=dataset_name, batch_size=batch_size_fid, dataset_res=32, num_gen=num_gen, dataset_split="train", mode="legacy_tensorflow")
    return score