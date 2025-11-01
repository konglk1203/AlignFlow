import torch
from .ctransform import ctransform


class SdotPlanSampler:
    """Helper class to evaluate SDOT map."""

    def __init__(self, data_tensor, entropic_reg=0):
        self.data_tensor = data_tensor
        self.data_tensor_flatten = data_tensor.reshape(data_tensor.shape[0], -1)
        self.dual_weight = None
        self.entropic_reg = entropic_reg

    def set_dual_weight(self, dual_weight):
        self.dual_weight = dual_weight

    def sample_plan(self, x0, x1, replace=True):
        r"""
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            this is useless;
        Returns
        -------
        x0[i] : randomly sampled
        x1[j] : data
        """
        entropic_reg = self.entropic_reg
        target_idx, hist, S_mat = ctransform(
            x0.reshape(x0.shape[0], -1),
            self.data_tensor_flatten,
            self.dual_weight,
            eps_entropic=self.entropic_reg,
        )
        if entropic_reg != 0:
            SM_S_mat = torch.nn.functional.softmax(-S_mat / entropic_reg, dim=1)
            target_idx = torch.vmap(
                lambda x: torch.multinomial(x, 1), randomness="different"
            )(SM_S_mat)

        target = self.data_tensor_flatten[target_idx].reshape(*x0.shape)
        return x0, target, target_idx
