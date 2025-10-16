# CIFAR-10 experiments on Unet

This folder is modified from [torchcfm](https://github.com/atong01/conditional-flow-matching).

## Usage
### Calculate the dual weight SDOT map
On a single GPU:
```bash
python sdot_compute_CIFAR10.py
```

On 8 GPUs:
```bash
python -m torch.distributed.launch --nproc_per_node=1 sdot_compute_CIFAR10.py
```
The dual weight we computed is provided in `CIFAR10_dual_weight_flip.pt`
### Train the flow matching models
The training will take around 20 hours on one L40S. Multi-GPU training will hurt the performance thus not supported.

Train with SDOT coupling
```bash
python train_cifar10.py --model AlignFlow --seed 0 --wandb_name AlignFlow_0
```

Train with minibatch OT
```bash
python train_cifar10.py --model otcfm --seed 0 --wandb_name otcfm_0
```

Vanilla FM
```bash
python train_cifar10.py --model icfm --seed 0 --wandb_name vanilla_fm_0
```

### Evaluate the FID score
```bash
torchrun --nproc-per-node=8 compute_fid_multi_gpu.py
```
This computes fid-50k for all checkpoints that matches the `path_pattern` on multiple GPUs. The results will saved to a csv file stated in `result_csv_path` argument. Please adjust these 2 parameters in the python code. 

Note: The fid logged by wandb is FID-10k for faster training, however, it will be worse than FID-50k.