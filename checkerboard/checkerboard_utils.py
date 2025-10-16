import torch
from torch import nn, Tensor

from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
)
from tqdm import tqdm

from torchdiffeq import odeint
import matplotlib.pyplot as plt
from matplotlib import cm

device = "cuda:0" if torch.cuda.is_available() else "cpu"
fm_dict = {
    "otfm": ExactOptimalTransportConditionalFlowMatcher,
    "vanillafm": ConditionalFlowMatcher,
    "sdotfm": ConditionalFlowMatcher,
}


# Generate checkerboard data
def generate_checkerboard_data(num_data: int = 200):
    x1 = torch.rand(num_data) * 4 - 2
    x2_ = torch.rand(num_data) - torch.randint(high=2, size=(num_data,)) * 2
    x2 = x2_ + (torch.floor(x1) % 2)

    data = 1.0 * torch.cat([x1[:, None], x2[:, None]], dim=1) / 0.45

    return data


# Activation class
class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return torch.sigmoid(x) * x


# Model class
class MLP(nn.Module):
    def __init__(self, input_dim: int = 2, time_dim: int = 1, hidden_dim: int = 128):
        super().__init__()

        self.input_dim = input_dim
        self.time_dim = time_dim
        self.hidden_dim = hidden_dim

        self.main = nn.Sequential(
            nn.Linear(input_dim + time_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        sz = x.size()
        x = x.reshape(-1, self.input_dim)
        t = t.reshape(-1, self.time_dim).float()

        t = t.reshape(-1, 1).expand(x.shape[0], 1)
        h = torch.cat([x, t], dim=1)
        output = self.main(h)

        return output.reshape(*sz)


class WrappedModel(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, t: torch.Tensor, x: torch.Tensor, **extras):
        return self.model(x, t)


def get_vector_field(method, model, dataloader, num_training_steps, lr):

    # init optimizer
    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    # train
    data_iter = iter(dataloader)

    FM = fm_dict[method](sigma=0)

    with tqdm(range(num_training_steps)) as pbar:
        for i in pbar:
            try:
                x_0, x_1 = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                x_0, x_1 = next(data_iter)
                x_1 = x_1.to(device)
                x_0 = x_0.to(device)
            optim.zero_grad()

            t, xt, ut = FM.sample_location_and_conditional_flow(x_0, x_1)
            vt = model(xt, t)
            loss = torch.mean((vt - ut) ** 2)
            pbar.set_postfix({"loss": loss.item()})

            loss.backward()
            optim.step()


def plot_density_traj(model, num_steps=10, num_samples=50000, save_path=None):
    norm = cm.colors.Normalize(vmax=50, vmin=0)
    t_span = torch.linspace(0, 1, num_steps).to(device=device)  # sample times

    x_init = torch.randn((num_samples, 2), dtype=torch.float32, device=device)
    traj = odeint(model, x_init, t_span, method="euler")  # create an ODESolver class

    traj = traj.detach().cpu().numpy()
    t_span = t_span.cpu()

    fig, axs = plt.subplots(1, num_steps, figsize=(20, 20))

    for i in range(10):
        H = axs[i].hist2d(traj[i, :, 0], traj[i, :, 1], 300, range=((-5, 5), (-5, 5)))

        cmin = 0.0
        cmax = torch.quantile(torch.from_numpy(H[0]), 0.99).item()

        norm = cm.colors.Normalize(vmax=cmax, vmin=cmin)

        _ = axs[i].hist2d(
            traj[i, :, 0],
            traj[i, :, 1],
            300,
            range=((-5, 5), (-5, 5)),
            norm=norm,
            rasterized=True,
        )

        axs[i].set_aspect("equal")
        axs[i].axis("off")
        axs[i].set_title("t= %.2f" % (t_span[i]))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()
