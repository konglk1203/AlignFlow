# AlignFlow: Improving Flow-based Generative Models with Semi-Discrete Optimal Transport

AlignFlow is a novel approach that leverages Semi-Discrete Optimal Transport (SDOT) to enhance the training of Flow-based Generative Models (FGMs) by establishing an explicit, optimal alignment between noise distribution and data points with guaranteed convergence. 

AlignFlow is a plug-and-play algorithm and can be easily applied to your own dataset and FGM algorithm without further hyperparameter tunning. It only adds minimal cost of computing.

This repository contains the 4 main experiments: 
| Experiment | Section | Space   | conditional | Combined with |
| :------- | :------- | :------- | :-------| :------- |
| Checkerboard     | Sec. D     | -             | - | Vanilla Flow matching |
| U-net + CIFAR-10 | Sec. 5.1   | pixel space   | unconditional | Vanilla Flow matching | 
| DiT on Imagenet  | Sec. 5.2   | latent space  | class-conditional | Shortcut model |
| SiT on Imagenet  | Sec. 5.3   | latent space  | class-conditional | MeanFlow |

The structure of this repository is:
```
AlignFlow
├── sdot                        # A package for computing and evaluating the SDOT map
│   ├── ctransform              # helper function for evaluating the SDOT map given the dual weight.
│   ├── generate_sdot_dataset   # generate (key, target_idx) pairs s.t. (randn(key=key), dataset[target_idx] is a data pair)
│   ├── sdot_plan_sampler       # Helper function for evaluating the SDOT map
│   ├── rebalance               # Rebalance the indices s.t. each data in the dataset is uniformly sampled
│   └── sdot_solve              # Given a dataset, compute the dual
├── checkerboard                # Synthetic test demonstrating the difference of SDOT coupling, random coupling and minibatch coupling. (Sec. D in the paper)
├── CIFAR                     # Train U-net (34M parameters) on CIFAR10. The flow matching is performed in the pixel space. (Sec. 5.1 in the paper)
├── imagenet_sdot_utils         # Helper functions for sdot dataset on Imagenet
├── shortcut-sdot               # Train DiT on Imagenet (256*256 resolution) by AlignFlow combining with vanilla flow matching, consistency training, live reflow and shortcut models. (Sec. 5.2 in the paper)
└── MeanFlow                    # Train SiT on Imagenet (256*256 resolution) by AlignFlow combining with meanflow. (Sec. 5.3 in the paper)
```

# How to use the code
Please make sure you are using python 3.9 and cuda 12. Then install packages by
```bash
pip install -r requirements.txt
```
Then add this folder containing this `README.md` file to path
```bash
export PYTHONPATH="${PYTHONPATH}:PATH_COTAINING_THIS_README_FILE" # For Linux
$env:PYTHONPATH = "$env:PYTHONPATH;PATH_COTAINING_THIS_README_FILE" # For Windows
```
- Please follow the README.md file in each folder to generate the result provided in the paper.

# When you have trouble running, the following hint may be helpful
If memory explodes or you have a deadlock, try change some of the following  environmental variables 
```bash
export TF_FORCE_GPU_ALLOW_GROWTH=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

