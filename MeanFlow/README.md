# AlignFlow + MeanFlow
This folder is modified from [pytorch implementation for shortcut models](https://github.com/zhuyu-cs/MeanFlow). It is not the official implementation but is able to reproduce the results in the paper.

## Usage
### download dataset
```bash
sudo apt-get install aria2 # wget also works
aria2c -x 16 https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar -d  ./preprocess_imagenet
aria2c -x 16 https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar -d  ./preprocess_imagenet
```

Extract the dataset and construct lmdb dataset by
```bash
cd ./preprocess_imagenet
sh untar.sh
python convert_lmdb.py
```

Save the VAE latent by
```bash
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
    main_cache.py \
    --source_lmdb /data/ImageNet_train \
    --target_lmdb /data/train_vae_latents_lmdb \
    --img_size 256 \
    --batch_size 1024 \
    --lmdb_size_gb 400
```


### config accelerate
```bash
accelerate config
```

### setup some constants 
```bash
export TEMP_DIR=YOUR_PATH/MeanFlow/temp # set temp dir path
export NUM_CLASSES_IMAGENET=1000
export MAX_NUM_EPOCH=250
```
### generate sdot dataset
```bash
python preprocess_save_vae.py
python ../imagenet_sdot_utils/preprocess_runner.py --task compute_sdot
python ../imagenet_sdot_utils/preprocess_runner.py --task generate_sdot_dataset
```

### Training
`--sdot` means use MeanFlow w/ Alignflow while `--no-sdot` will use MeanFlow w/0 Alignflow

The following commands is recommend to run on 2 or 4 GPUs
```bash
# MeanFlow w/ Alignflow on SiT-B/4
accelerate launch --multi_gpu --num_processes 2 --gpu_ids 2,3 \
    train.py \
    --exp-name "meanflow_b_4_sdot_new" \
    --model "SiT-B/4" \
    --epochs 80\
    --cfg-omega 3.0 \
    --cfg-kappa 0.\
    --cfg-min-t 0.0\
    --cfg-max-t 1.0 \
    --sdot

# MeanFlow w/o Alignflow on SiT-B/4
accelerate launch --multi_gpu --num_processes 2 --gpu_ids 6,7 \
    train.py \
    --exp-name "meanflow_b_4_nosdot" \
    --model "SiT-B/4" \
    --epochs 80\
    --cfg-omega 3.0 \
    --cfg-kappa 0.\
    --cfg-min-t 0.0\
    --cfg-max-t 1.0 \
    --no-sdot
```


The following commands is recommend to run on 4 or 8 GPUs (100 hours on 4*H100; 70 hours on 8*H100)
```bash
# MeanFlow w/ Alignflow on SiT-B/2
accelerate launch --multi_gpu --num_processes 4 --gpu_ids 0,1,2,3 \
accelerate launch --multi_gpu --num_processes 8 \
    train.py \
    --exp-name "meanflow_b_2_sdot" \
    --model "SiT-B/2" \
    --epochs 240\
    --cfg-omega 1.0 \
    --cfg-kappa 0.5\
    --cfg-min-t 0.0\
    --cfg-max-t 1.0\
    --sdot \
    --data-dir /home/greenland-user/MeanFlow/data/ImageNet/train_vae_latents_lmdb \
    --resume-step 0600000

# MeanFlow w/o Alignflow on SiT-B/2
accelerate launch --multi_gpu  --num_processes 4 --gpu_ids 4,5,6,7\
    train.py \
    --exp-name "meanflow_b_2_nosdot" \
    --model "SiT-B/2" \
    --epochs 240\
    --cfg-omega 1.0 \
    --cfg-kappa 0.5\
    --cfg-min-t 0.0\
    --cfg-max-t 1.0\
    --no-sdot\
    --output-dir /tmp/instance_storage/results/ \
    --data-dir /home/greenland-user/MeanFlow/data/ImageNet/train_vae_latents_lmdb 
```

```bash
# MeanFlow w/ Alignflow on SiT-L/2
accelerate launch --multi_gpu  --num_processes 8\
    train.py \
    --exp-name "meanflow_l_2_sdot" \
    --model "SiT-L/2" \
    --epochs 240\
    --cfg-omega 0.2 \
    --cfg-kappa 0.92\
    --cfg-min-t 0.0\
    --cfg-max-t 0.8\
    --sdot \
    --output-dir /tmp/instance_storage/results/ \
    --data-dir /home/greenland-user/MeanFlow/data/ImageNet/train_vae_latents_lmdb \
    --resume-step 1050000 
# MeanFlow w/o Alignflow on SiT-L/2
accelerate launch --multi_gpu  --num_processes 8 \
    train.py \
    --exp-name "meanflow_l_2_nosdot" \
    --model "SiT-L/2" \
    --epochs 240\
    --cfg-omega 0.2 \
    --cfg-kappa 0.92\
    --cfg-min-t 0.0\
    --cfg-max-t 0.8
    --no-sdot
```


On 8*H200, the following commands needs 180 hours
```bash
# MeanFlow w/o Alignflow on SiT-XL/2
accelerate launch --multi_gpu  --num_processes 8\
    train.py \
    --exp-name "meanflow_xl_2_sdot"\
    --model "SiT-XL/2" \
    --epochs 240\
    --cfg-omega 0.2 \
    --cfg-kappa 0.92\
    --cfg-min-t 0.0\
    --cfg-max-t 0.75 \
    --sdot \
    --output-dir /tmp/instance_storage/results/ \
    --data-dir /home/greenland-user/MeanFlow/data/ImageNet/train_vae_latents_lmdb \
    --resume-step 1100000

# MeanFlow w/o Alignflow on SiT-XL/2
accelerate launch --multi_gpu  --num_processes 8\
    train.py \
    --exp-name "meanflow_xl_2_nosdot" \
    --model "SiT-XL/2" \
    --epochs 240\
    --cfg-omega 0.2 \
    --cfg-kappa 0.92\
    --cfg-min-t 0.0\
    --cfg-max-t 0.75 \
    --no-sdot \
    --output-dir /tmp/instance_storage/results/ \
    --data-dir /home/greenland-user/MeanFlow/data/ImageNet/train_vae_latents_lmdb
```


### Evaluation
For evaluating a checkpoint, run
```bash
torchrun --nproc_per_node=4 evaluate.py \
    --ckpt "./results/meanflow_l_2_sdot/checkpoints/0650000.pt" \
    --model "SiT-L/2" \
    --cfg-scale 1.0 \
    --per-proc-batch-size 128 \
    --num-fid-samples 50000 \
    --sample-dir "./fid_dir" \
    --compute-metrics \
    --num-steps 1\
    --fid-statistics-file "./fid_stats/adm_in256_stats.npz"
```


### Notes
- When evaluating models trained with CFG , the --cfg-scale parameter must be set to 1.0 during inference, as the CFG guidance has been incorporated into the model during training and is no longer controllable at sampling time.

- Due to we are using unofficial code, we are using a slightly different VAE from the paper, see [this issue](https://github.com/zhuyu-cs/MeanFlow/issues/7). However, we are able to reproduce similar results with the paper.

- If you have error:
/ubuntu/code/src/torch-fidelity/torch_fidelity/metric_fid.py", line 35, 
`AssertionError: assert mu1.ndim == 1 and mu1.shape == mu2.shape and mu1.dtype == mu2.dtype` in function `fid_statistics_to_metric`

Please add:
```python
    mu1=mu1.astype(np.float32)
    mu2=mu2.astype(np.float32)
    sigma1=sigma1.astype(np.float32)
    sigma2=sigma2.astype(np.float32)
```
following 
```python
mu1, sigma1 = stat_1["mu"], stat_1["sigma"]
mu2, sigma2 = stat_2["mu"], stat_2["sigma"]
```
in file `torch-fidelity/torch_fidelity/metric_fid.py`


### Checkpoints
The checkpoints for the models can be downloaded from google drive:
./results/meanflow_b_4_sdot/checkpoints/0400000.pt
./results/meanflow_b_2_sdot/checkpoints/1200000.pt
./results/meanflow_l_2_sdot/checkpoints/1100000.pt
./results/meanflow_xl_2_sdot/checkpoints/1100000.pt