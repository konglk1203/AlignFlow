import torch
from torch import Tensor
import torch.distributed as dist
from sdot import ctransform
from tqdm import tqdm
import os


@torch.no_grad()
def sdot_solve(
    targets: Tensor,
    nu=None,
    lr=1,
    num_step=2_000,
    batch_size=128,
    dual_weight_init=None,
    eps_entropic=0.01,
    filename=None,
    save_every=None,
    ema_param=0.9999,
    pbar_args={"desc": None},
    use_multi_gpu=False,
    info_dict=None,
):
    """
    This function computes the sdot map from standard Gaussion to targets.
    Parameters:
        targets: Required. Tensor with shape [num_target, dim].
        nu: Tensor with shape [num_target, ], as the weight of targets. If None, uniform weight will be used.
        lr: learning rate for the optimization (Adam optimizer will be used).
        num_step: number of optimization steps.
        batch_size: number of samples each step.
        dual_weight_init: initial value for dual_weight. 0 initialization will be used if not provided.
        eps_entropic: entropic regularization strength. Should be a positive value or 0.
        filename: the dual_weight will be saved to this file.
        save_every: interval for saveing dual_weight.
        ema_param: we use ema to stablize the training process.
        use_multi_gpu: Whether to use multiple GPUs for computation. For multi-GPU, set this value to True and use command `python -m torch.distributed.launch --nproc_per_node=4 your_file_using_this_function.py`
    Return:
        dual_weight_ema: the dual weight to be used later in computing the sdot map
        info_dict: contains the training curve
    Example usage:
        Please see sdot-fm/sdot_compute_CIFAR10.py for an example.
    """
    # Initialize distributed environment if using multi-GPU
    if use_multi_gpu:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = targets.device
        local_rank = 0
        world_size = 1

    # Move targets to the appropriate device
    targets = targets.to(device)

    # targets: discrete distribution
    # nu: weight of targets. None means uniform weight
    num_target, dim = targets.shape

    if nu is None:
        nu = torch.ones(num_target, device=device) / num_target
    else:
        nu = nu.to(device)
        assert nu.shape[0] == num_target, "size of Y and nu does not match"
        nu_min = float(nu.min().item())
        assert nu_min > 0.0, "nu minimum value not greater than 0"

    if dual_weight_init is None:
        dual_weight_init = torch.zeros(num_target, device=device)
    else:
        dual_weight_init = dual_weight_init.to(device)

    dual_weight = dual_weight_init.clone()

    # When using multi-GPU, we split the batch across GPUs
    local_batch_size = batch_size // world_size if use_multi_gpu else batch_size
    X = torch.zeros((local_batch_size, dim), device=device)

    optimizer = torch.optim.Adam([dual_weight], lr=lr, amsgrad=True)

    # Only show progress bar on rank 0
    if local_rank == 0:
        pbar = tqdm(range(num_step), **pbar_args)
    else:
        pbar = range(num_step)
    if info_dict == None:
        dual_weight_grad_ema = 0
        mre_list = []
        L1_list = []
    else:
        dual_weight_grad_ema = info_dict["dual_weight_grad_ema"].to(device)
        mre_list = info_dict["mre_list"]
        L1_list = info_dict["L1_list"]

    dual_weight_ema = dual_weight.detach()

    for k in pbar:
        X = torch.randn_like(X)
        target_idx, hist, _ = ctransform(
            X, targets, dual_weight, eps_entropic=eps_entropic
        )

        # Synchronize histograms across all GPUs when using multi-GPU
        if use_multi_gpu:
            dist.all_reduce(hist, op=dist.ReduceOp.AVG)
        dual_weight.grad = hist - nu

        if (
            save_every is not None
            and filename is not None
            and (k % save_every == 0 or k == num_step - 1)
        ):
            # Save only from rank 0
            if local_rank == 0:
                torch.save(dual_weight_ema, filename)
            continue

        optimizer.step()

        # This is correct. Line 10, Algo 1 in the paper seems incorrect.
        dual_weight_grad_ema = (
            ema_param * dual_weight_grad_ema + (1 - ema_param) * dual_weight.grad
        )

        mre_est = float(torch.max(torch.abs(dual_weight_grad_ema) / nu).item())
        L1_est = float(torch.norm(dual_weight_grad_ema, p=1.0).item())

        # Only update progress bar on rank 0
        if local_rank == 0:
            log_dict = {"mre_est": mre_est, "L1_est": L1_est}
            if isinstance(pbar, tqdm):
                pbar.set_postfix(log_dict)

        dual_weight_ema = (
            ema_param * dual_weight_ema + (1 - ema_param) * dual_weight.detach()
        )

        mre_list.append(mre_est)
        L1_list.append(L1_est)

    # Return results
    info_dict = {
        "mre_list": mre_list,
        "L1_list": L1_list,
        "dual_weight_grad_ema": dual_weight_grad_ema,
    }
    return dual_weight_ema, info_dict


# How to Use:
# Run with multiple GPUs using PyTorch's distributed launch utility:
# python -m torch.distributed.launch --nproc_per_node=4 your_script.py
# In your main script, call the function with use_multi_gpu=True:
# dual_weight, info = sdot_solve(targets, use_multi_gpu=True, ...)
