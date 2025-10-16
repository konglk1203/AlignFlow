# Checkerboard plot

1. add the folder containing `sdot-fm` to path, e.g.,
```bash
export PYTHONPATH=$PYTHONPATH:/home/ubuntu/code/sdot-fm
```

2. run the following commands to generate the figures in Sec. ??
```bash
CUDA_VISIBLE_DEVICES=0 python checkerboard.py --method sdotfm --batch_size 4096 --save_fig_path checkerboard_sdotfm.jpg

CUDA_VISIBLE_DEVICES=1 python checkerboard.py --method vanillafm --batch_size 4096 --save_fig_path checkerboard_vanillafm.jpg

CUDA_VISIBLE_DEVICES=2 python checkerboard.py --method otfm --batch_size 128 --save_fig_path checkerboard_otfm.jpg --num_training_steps 200000
```