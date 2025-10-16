import torch
from checkerboard_utils import (
    MLP,
    generate_checkerboard_data,
    get_vector_field,
    WrappedModel,
    plot_density_traj,
)
from torch.utils.data import Dataset, DataLoader, default_collate

device = "cuda:0" if torch.cuda.is_available() else "cpu"


import argparse

parser = argparse.ArgumentParser(description="A simple script to greet a user.")
parser.add_argument("--method", type=str, default="sdotfm")
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--batch_size", type=int, default=4096)
parser.add_argument("--hidden_dim", type=int, default=512)
parser.add_argument("--num_training_steps", type=int, default=5000)
parser.add_argument("--num_data", type=int, default=2000)
parser.add_argument("--save_fig_path", type=str, default=None)

args = parser.parse_args()

# compute SDOT map

data_tensor = generate_checkerboard_data(num_data=args.num_data).cuda()

if args.method == "sdotfm":
    from sdot import SdotPlanSampler
    from sdot_rebalance_dataset import SdotRebalanceDataset

    ot_sampler = SdotPlanSampler(data_tensor)
    from sdot import sdot_solve

    dual_weight, _ = sdot_solve(
        data_tensor,
        lr=0.1,
        num_step=1000,
        eps_entropic=0.01,
        batch_size=1024,
        ema_param=0.99,
    )

    ot_sampler.set_dual_weight(dual_weight)

    sdot_ds = SdotRebalanceDataset(ot_sampler, 10 * len(data_tensor))
    sdot_ds.refresh()

    def collate_fn(batch):
        x_0, x_1_idx = default_collate(batch)
        x_1 = data_tensor[x_1_idx]
        x_1 = x_1.to(device)
        x_0 = x_0.to(device)
        return x_0, x_1

    dataloader = DataLoader(
        sdot_ds,
        batch_size=args.batch_size,
        drop_last=True,
        shuffle=True,
        collate_fn=collate_fn,
    )
elif args.method in ["vanillafm", "otfm"]:

    class RandomNumberDataset(Dataset):
        def __init__(self):
            super().__init__()

        def __len__(self):
            return args.batch_size

        def __getitem__(self, index):
            return torch.randint(high=len(data_tensor), size=(1,)).to(device)

    ds = RandomNumberDataset()

    def collate_fn(batch):
        x_1_idx = default_collate(batch)
        x_1 = data_tensor[x_1_idx]
        x_1 = x_1.to(device)
        x_0 = torch.randn_like(x_1)
        return x_0, x_1

    dataloader = DataLoader(ds, batch_size=args.batch_size, collate_fn=collate_fn)

else:
    raise

model = MLP(input_dim=2, time_dim=1, hidden_dim=args.hidden_dim).to(device)

vf = get_vector_field(args.method, model, dataloader, args.num_training_steps, args.lr)
model = WrappedModel(model)

plot_density_traj(model, save_path=args.save_fig_path)
