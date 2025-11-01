import torch
from torch import Tensor
from typing import Tuple


def get_cost_matrix(X: Tensor, Y: Tensor) -> Tensor:
    """
    This function computes the (x-y)^2
    Parameters:
        X: noise. Shape: [num_noise, dim].
        Y: targets. Shape: [num_targets, dim].
    Return:
        cost: Shape: [num_noise, num_targets].
    """

    assert X.shape[1] == Y.shape[1], "data size does not match"
    nX = X.shape[0]
    nY = Y.shape[0]
    if nX == 1 or nY == 1:
        diff = X - Y
        cost = (diff * diff).sum(1, keepdim=True)
        if nX == 1:
            return torch.transpose(cost, 0, 1)
        else:
            return cost
    else:
        Y = torch.transpose(Y, 0, 1)
        X2 = (X * X).sum(1, keepdim=True)
        Y2 = (Y * Y).sum(0, keepdim=True)
        XY = torch.mm(X, Y)
        cost = torch.clamp(X2.expand(nX, nY) + Y2.expand(nX, nY) - 2.0 * XY, min=0.0)
        return cost


def ctransform(
    X: Tensor, Y: Tensor, v: Tensor, coef: float = 1.0, eps_entropic: float = 0
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    This function computes the ctransform, will be used to evaluate the sdot map. see Eq. ?? in the paper.
    Parameters:
        X: noise. Shape: [num_noise, dim].
        Y: targets. Shape: [num_targets, dim].
        eps_entropic: when non-zero, target index will be sampled from the softmax distribution.
    Return:
        target_idx: the target index for each noise. Shape: [num_noise,].
        hist: the histogram of the target index. Shape: [num_targets,]. Will be used to compute gradient.
        S: the cost matrix minus dual weight.
    Note:
        Setting non-zero eps_entropic will lead to randomness in the matching between noise and targets.
    """
    assert X.shape[1] == Y.shape[1], "data size does not match"
    nX = X.shape[0]
    nY = Y.shape[0]

    C = get_cost_matrix(X, Y) * coef
    S = C - v
    if eps_entropic == 0:
        val, target_idx = torch.min(S, 1)
        hist = torch.histc(target_idx, bins=nY, min=0, max=nY - 1)
        hist = hist / hist.sum()
    else:
        _, target_idx = torch.min(S, 1)
        SM_mat = torch.nn.functional.softmax(-S / eps_entropic, dim=1)
        hist = torch.mean(SM_mat, dim=0)
        target_idx = torch.vmap(
            lambda x: torch.multinomial(x, 1), randomness="different"
        )(SM_mat)
    return target_idx, hist, S