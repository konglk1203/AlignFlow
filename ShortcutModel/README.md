## AlignFlow + Chortcut Model
In this code, we show AlignFlow can improves the performance for [shortcut models](https://arxiv.org/abs/2410.12557), which is one of the SOTA Flow-based Generative Model algorithms.

This folder is modified from [official implementation for shortcut models](https://github.com/kvfrans/shortcut-models).


## Usage
### setup some constants 
```bash
export TFDS_DATA_DIR=YOUR_PATH/tensorflow_datasets # set path to tensorflow dataset
export TEMP_DIR=YOUR_PATH/AlignFlow/ShortcutModel/temp # set temp dir path
export NUM_CLASSES_IMAGENET=1000
export MAX_NUM_EPOCH=150
```
### download dataset
```bash
sudo apt-get install aria2 # wget also works
aria2c -x 16 https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar -d  ${TFDS_DATA_DIR}/downloads/manual
aria2c -x 16 https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar -d  ${TFDS_DATA_DIR}/downloads/manual
```
Please do not unzip the tar files since it will be extracted automatically in the next step.
### compute sdot map and generate sdot dataset
The following commands saves vae latents, compute the sdot map in the latent space and generate the sdot datasets.
```bash
python preprocess_save_vae_tf.py
python ../imagenet_sdot_utils/preprocess_runner.py --task compute_sdot
python ../imagenet_sdot_utils/preprocess_runner.py --task generate_sdot_dataset
```
### training
DiT-B on Imagenet-256 (All the following commands runs on 2 or 4 L40S gpus. 8 GPUs will be unnecessary and inefficient); `--sdot` means using AlignFlow, while `--nosdot` means using independent coupling.
```bash
python train_latent.py --model.train_type shortcut --sdot --max_steps 2010000
python train_latent.py --model.train_type shortcut --nosdot --max_steps 2010000

python train_latent.py --model.train_type progressive --sdot 
python train_latent.py --model.train_type progressive --nosdot

python train_latent.py --model.train_type livereflow --sdot
python train_latent.py --model.train_type livereflow --nosdot

python train_latent.py --model.train_type consistency --sdot
python train_latent.py --model.train_type consistency --nosdot

python train_latent.py --model.train_type naive --sdot
python train_latent.py --model.train_type naive --nosdot
```

### evaluate the model
To validate all checkpoints in some folder, please run
```bash
python validate_all.py --checkpoint_path "./results/**/1300001.tmp" --inference_timesteps 4 --batch_size 256
```
This commands generates samples and computes fid-50k for all checkpoints that matches the `checkpoint_path` argument (Do not forget the double quote at path). The results will be generated in the same folder with `*.tmp` file:
- `info.csv` contains the fid-50k score with different number of steps
- `x_render_step_*.npy` contains sample images with given number of integration steps and `lab_step_*.npy` contains the label index for the generated images.

Note: Vae may return different value on different number of GPU and different batch size. This does not indicate a bug.

The checkpoints for the models can be downloaded from google drive:
./results/shortcut_sdot/1300001/1300001.pt