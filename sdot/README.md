# SDOT solver
This folder contains SDOT solver and can be included as a package. The package `sdot` contains the following functions:

- `ctransform`: Given a noise, match the noise to a point in the dataset by the semi-discrete optimal transport map.
- `sdot_solve`: Given a dataset, solve the SDOT map from element-wise iid Gaussian to the dataset and return dual weight.
- `SdotPlanSampler`: Evaluate SDOT map given dual weight.
- `rebalance`: Given an input index list, this function returns a new list s.t. each index has same number of occurrence with minimum change of the list. See Sec. B in the paper.
- `generate_sdot_dataset`: This function generates a sdot dataset by returning the keys for noise and the target index in the original dataset. Using the function 
- `generate_tensor_from_seed`: recover the noise from random seed in the dataset.
- `generate_tensor_from_seeds_vmap`: vectorized version for `generate_tensor_from_seed`.