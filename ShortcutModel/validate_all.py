import jax.numpy as jnp
from absl import app, flags
import numpy as np
import jax
import jax.numpy as jnp
from ml_collections import config_flags
import ml_collections
import json
from utils.train_state import TrainStateEma
from utils.checkpoint import Checkpoint
from utils.stable_vae import StableVAE
from utils.sharding import create_sharding
from model import DiT
from helper_inference import do_inference
import torch
from torch.utils.data import default_collate
from jax.tree_util import tree_map
from my_utils import InfiniteDataLoader
from imagenet_sdot_utils.sdot_latent_dataset import ValidateDatasetLatent
from glob import glob
import pandas as pd
import os
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset_name', 'imagenet256_latent', 'Environment name.')
flags.DEFINE_string('load_dir', None, 'If not None, load params from checkpoint.')
flags.DEFINE_string('save_dir', None, 'Save checkpoints to this path.')
flags.DEFINE_string('fid_stats', 'data/imagenet256_fidstats_ours.npz', 'FID stats file.')
flags.DEFINE_integer('seed', 10, 'Random seed.') # Must be the same across all processes.
flags.DEFINE_integer('batch_size', 256, 'Minibatch size.')
flags.DEFINE_integer('max_steps', int(1_000_000), 'Number of training steps.')
flags.DEFINE_integer('debug_overfit', 0, 'Debug overfitting.')
flags.DEFINE_string('checkpoint_path', './results_XL/**/*.tmp', 'All checkpoints in this pattern will be evaluated.')

# please switch to the correct model config file

with open("DiT_B_default.json") as json_file:
# with open("DiT_XL_default.json") as json_file:
    json_data = json.load(json_file)
    print(json_data)
model_config = ml_collections.ConfigDict(json_data)
config_flags.DEFINE_config_dict('model', model_config, lock_config=False)


##############################################
## Training Code.
##############################################
def main(_):
    np.random.seed(FLAGS.seed)
    print("Using devices", jax.local_devices())
    device_count = len(jax.local_devices())
    global_device_count = jax.device_count()
    print("Device count", device_count)
    print("Global device count", global_device_count)
    local_batch_size = FLAGS.batch_size // (global_device_count // device_count)
    print("Global Batch: ", FLAGS.batch_size)
    print("Node Batch: ", local_batch_size)
    print("Device Batch:", local_batch_size // device_count)

    def numpy_collate(batch):
        return tree_map(np.asarray, default_collate(batch))

    val_dataset_latent = ValidateDatasetLatent()
    dataset_valid = torch.utils.data.DataLoader(
        val_dataset_latent,
        batch_size=FLAGS.batch_size,
        num_workers=4,
        drop_last=True,
        collate_fn=numpy_collate,
        shuffle=False,
    )
    dataset_valid = InfiniteDataLoader(dataset_valid)

    example_obs, example_labels = next(dataset_valid)
    example_obs = example_obs[:1]
    example_obs_shape = example_obs.shape

    if FLAGS.model.use_stable_vae:
        vae = StableVAE.create()
        if 'latent' in FLAGS.dataset_name:
            example_obs = example_obs[:, :, :, example_obs.shape[-1] // 2:]
            example_obs_shape = example_obs.shape
        else:
            example_obs = vae.encode(jax.random.PRNGKey(0), example_obs)
        example_obs_shape=(1, 32,32,4)
        example_obs = jnp.zeros(example_obs_shape)
        vae_rng = jax.random.PRNGKey(42)
        vae_encode = None
        vae_decode = jax.jit(vae.decode)

    from utils.fid import get_fid_network, fid_from_stats
    get_fid_activations = get_fid_network() 
    truth_fid_stats = np.load(FLAGS.fid_stats)

    ###################################
    # Creating Model and put on devices.
    ###################################
    FLAGS.model.image_channels = example_obs_shape[-1]
    FLAGS.model.image_size = example_obs_shape[1]
    dit_args = {
        'patch_size': FLAGS.model['patch_size'],
        'hidden_size': FLAGS.model['hidden_size'],
        'depth': FLAGS.model['depth'],
        'num_heads': FLAGS.model['num_heads'],
        'mlp_ratio': FLAGS.model['mlp_ratio'],
        'out_channels': example_obs_shape[-1],
        'class_dropout_prob': FLAGS.model['class_dropout_prob'],
        'num_classes': FLAGS.model['num_classes'],
        'dropout': FLAGS.model['dropout'],
        'ignore_dt': False if (FLAGS.model['train_type'] in ('shortcut', 'livereflow')) else True,
    }
    model_def = DiT(**dit_args)

    def init(rng):
        param_key, dropout_key, dropout2_key = jax.random.split(rng, 3)
        example_t = jnp.zeros((1,))
        example_dt = jnp.zeros((1,))
        example_label = jnp.zeros((1,), dtype=jnp.int32)
        example_obs = jnp.zeros(example_obs_shape)
        model_rngs = {'params': param_key, 'label_dropout': dropout_key, 'dropout': dropout2_key}
        params = model_def.init(model_rngs, example_obs, example_t, example_dt, example_label)['params']
        return TrainStateEma.create(model_def, params, rng=rng)

    rng = jax.random.PRNGKey(FLAGS.seed)
    train_state_shape = jax.eval_shape(init, rng)

    data_sharding, train_state_sharding, no_shard, shard_data, global_to_local = (
        create_sharding(FLAGS.model.sharding, train_state_shape)
    )
    train_state = jax.jit(init, out_shardings=train_state_sharding)(rng)
    jax.debug.visualize_array_sharding(
        train_state.params["FinalLayer_0"]["Dense_0"]["kernel"]
    )
    jax.debug.visualize_array_sharding(
        train_state.params["TimestepEmbedder_1"]["Dense_0"]["kernel"]
    )
    jax.experimental.multihost_utils.assert_equal(
        train_state.params["TimestepEmbedder_1"]["Dense_0"]["kernel"]
    )

    path_list = glob(FLAGS.checkpoint_path, recursive=True)

    for path in path_list:
        print(path)
        FLAGS.save_dir = os.path.dirname(path)
        FLAGS.load_dir = path
        save_dir = FLAGS.save_dir
        result_csv_path = os.path.join(save_dir, "info.csv")
        if not os.path.exists(result_csv_path):
            column_names = ["step", "fid-50k"]
            df = pd.DataFrame(columns=column_names)
            df.to_csv(result_csv_path, index=False)
        df = pd.read_csv(result_csv_path, header=0)
        if FLAGS.inference_timesteps in list(df["step"]):
            continue

        cp = Checkpoint(FLAGS.load_dir)
        replace_dict = cp.load_as_dict()["train_state"]
        del replace_dict["opt_state"]  # Debug
        train_state = train_state.replace(**replace_dict)
        train_state = jax.jit(lambda x: x, out_shardings=train_state_sharding)(
            train_state
        )
        print("Loaded model with step", train_state.step)
        train_state = train_state.replace(step=0)
        del cp

        # refresh dataloader
        dataset_valid = InfiniteDataLoader(dataset_valid)
        _ = next(dataset_valid)

        fid_score = do_inference(
            FLAGS,
            train_state,
            dataset_valid,
            shard_data,
            vae_encode,
            vae_decode,
            get_fid_activations,
            fid_from_stats,
            truth_fid_stats,
        )
        if jax.process_index() == 0:

            new_row_data = {"step": FLAGS.inference_timesteps, "fid-50k": fid_score}
            new_row_df = pd.DataFrame([new_row_data])
            new_row_df.to_csv(result_csv_path, mode="a", header=False, index=False)


if __name__ == "__main__":
    app.run(main)
