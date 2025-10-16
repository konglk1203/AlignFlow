import copy
import os
import wandb

import torch
from absl import app, flags
from torchvision import datasets, transforms
from tqdm import trange
from utils_cifar import ema, generate_samples
from torchvision.utils import make_grid

from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    TargetConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher,
)
from torchcfm.models.unet.unet import UNetModelWrapper
from compute_fid_utils import get_fid

FLAGS = flags.FLAGS

flags.DEFINE_string("model", "AlignFlow", help="flow matching model type")
flags.DEFINE_string("output_dir", "./results/", help="output_directory")
flags.DEFINE_string("wandb_name", 'debug', help="wandb experiment name")

# UNet
flags.DEFINE_integer("num_channel", 128, help="base channel of UNet")

# Training
flags.DEFINE_float("lr", 2e-4, help="target learning rate")  # TRY 2e-4
flags.DEFINE_float("grad_clip", 1.0, help="gradient norm clipping")
flags.DEFINE_integer(
    "total_steps", 400001, help="total training steps"
)  # Lipman et al uses 400k but double batch size
flags.DEFINE_integer("warmup", 5000, help="learning rate warmup")
flags.DEFINE_integer("batch_size", 128, help="batch size")  # Lipman et al uses 128
flags.DEFINE_integer("num_workers", 4, help="workers of Dataloader")
flags.DEFINE_float("ema_decay", 0.9999, help="ema decay rate")
flags.DEFINE_bool("parallel", False, help="multi gpu training")
flags.DEFINE_float("sigma", 0.0, help="do not tune this. This value should always be 0.")
flags.DEFINE_integer("seed", None, help="random seed")

# for sdot only
flags.DEFINE_float("entropic_reg", 0, help="entropic regularization strength. We recommand using the same value as when calculating the dual weigth")
flags.DEFINE_integer("resample_interval", 10, help="number of epoch for each rebalance")

# Evaluation
flags.DEFINE_integer(
    "save_step",
    20000,
    help="frequency of saving checkpoints, 0 to disable during training",
)


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def warmup_lr(step):
    return min(step, FLAGS.warmup) / FLAGS.warmup


def train(argv):
    if FLAGS.seed!=None:
        torch.manual_seed(FLAGS.seed)
    print(
        "lr, total_steps, ema decay, save_step:",
        FLAGS.lr,
        FLAGS.total_steps,
        FLAGS.ema_decay,
        FLAGS.save_step,
    )

    run = wandb.init(
        project='AlignFlow-cifar',
        name= FLAGS.wandb_name,
        # name= 'debug',
        config=FLAGS
    )

    # DATASETS/DATALOADER

    if FLAGS.model in ["otcfm", "icfm", "fm", "si"]:
        dataset = datasets.CIFAR10(
            root="./data",
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=FLAGS.batch_size,
            shuffle=True,
            num_workers=FLAGS.num_workers,
            drop_last=True,
        )

        loader_iter = iter(dataloader)
    elif FLAGS.model=='AlignFlow':
        from sdot import SdotPlanSampler
        from sdot_rebalance_dataset import SdotRebalanceDataset
        
        data_tensor=torch.load('./pickled_cifar10_augmented.pt', map_location='cpu').cuda()
        data_tensor_flatten=data_tensor.reshape(data_tensor.shape[0], -1)


        ot_sampler=SdotPlanSampler(data_tensor_flatten, entropic_reg=FLAGS.entropic_reg)
        dual_weight=torch.load('./CIFAR10_dual_weight_flip.pt', map_location='cpu').cuda()
        ot_sampler.set_dual_weight(dual_weight)
        sdot_ds=SdotRebalanceDataset(ot_sampler, FLAGS.resample_interval*len(data_tensor))
        sdot_ds.refresh()

        

        dataloader = torch.utils.data.DataLoader(
            sdot_ds,
            batch_size=FLAGS.batch_size,
            num_workers=FLAGS.num_workers,
            drop_last=True,
            shuffle=True
        )
        loader_iter = iter(dataloader)

    else:
        raise NotImplementedError()

    

    #################################
    #            OT-CFM
    #################################

    sigma = 0.0
    if FLAGS.model=='AlignFlow':
        FM = ConditionalFlowMatcher(sigma=FLAGS.sigma)
    elif FLAGS.model == "otcfm":
        FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
    elif FLAGS.model == "icfm":
        FM = ConditionalFlowMatcher(sigma=sigma)
    elif FLAGS.model == "fm":
        FM = TargetConditionalFlowMatcher(sigma=sigma)
    elif FLAGS.model == "si":
        FM = VariancePreservingConditionalFlowMatcher(sigma=sigma)
    else:
        raise NotImplementedError(
            f"Unknown model {FLAGS.model}, must be one of ['otcfm', 'icfm', 'fm', 'si']"
        )

    savedir = os.path.join(FLAGS.output_dir, FLAGS.wandb_name)
    os.makedirs(savedir, exist_ok=True)

    # MODELS
    net_model = UNetModelWrapper(
        dim=(3, 32, 32),
        num_res_blocks=2,
        num_channels=FLAGS.num_channel,
        channel_mult=[1, 2, 2, 2],
        num_heads=4,
        num_head_channels=64,
        attention_resolutions="16",
        dropout=0.1,
    ).to(
        device
    )  # new dropout + bs of 128

    ema_model = copy.deepcopy(net_model)
    optim = torch.optim.Adam(net_model.parameters(), lr=FLAGS.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)
    if FLAGS.parallel:
        print(
            "Warning: parallel training is performing slightly worse than single GPU training due to statistics computation in dataparallel. We recommend to train over a single GPU, which requires around 8 Gb of GPU memory."
        )
        net_model = torch.nn.DataParallel(net_model)
        ema_model = torch.nn.DataParallel(ema_model)

    # show model size
    model_size = 0
    for param in net_model.parameters():
        model_size += param.data.nelement()
    print("Model params: %.2f M" % (model_size / 1024 / 1024))

    with trange(1, FLAGS.total_steps, dynamic_ncols=True) as pbar:
        for step in pbar:
            optim.zero_grad()
            try:
                batch = next(loader_iter)
            except StopIteration:
                if FLAGS.model=='AlignFlow':
                    dataloader.dataset.refresh()
                loader_iter = iter(dataloader) # Reset the iterator
                batch = next(loader_iter)
            if FLAGS.model=='AlignFlow':
                x0, target_idx=batch
                x1 = data_tensor[target_idx].cuda()
                x0 = x0.cuda().reshape(*x1.shape)
            else:
                x1, _=batch
                x1=x1.cuda()
                x0 = torch.randn_like(x1)
            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
            vt = net_model(t, xt)
            loss = torch.mean((vt - ut) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net_model.parameters(), FLAGS.grad_clip)  # new
            optim.step()
            sched.step()
            ema(net_model, ema_model, FLAGS.ema_decay)  # new
            run.log({"loss": loss.item()}, step=step)

            # sample and Saving the weights
            if FLAGS.save_step > 0 and step % FLAGS.save_step == 0:
                net_generated_img=generate_samples(net_model, FLAGS.parallel, savedir, step, net_="normal")
                ema_generated_img=generate_samples(ema_model, FLAGS.parallel, savedir, step, net_="ema")
                wandb.log({"net_generated_img": wandb.Image(make_grid(net_generated_img))}, step=step)
                wandb.log({"ema_generated_img": wandb.Image(make_grid(ema_generated_img))}, step=step)
                torch.save(
                    {
                        "net_model": net_model.state_dict(),
                        "ema_model": ema_model.state_dict(),
                        "sched": sched.state_dict(),
                        "optim": optim.state_dict(),
                        "step": step,
                    },
                    os.path.join(savedir, f"{FLAGS.model}_cifar10_weights_step_{step}.pt")
                )
                ema_model.eval()
                score=get_fid(ema_model)
                run.log({"fid": score}, step=step)
                ema_model.train()
                net_model.train()


if __name__ == "__main__":
    app.run(train)