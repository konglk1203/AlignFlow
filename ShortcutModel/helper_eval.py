import jax
import jax.experimental
import wandb
import jax.numpy as jnp
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from functools import partial
from sdot import generate_tensor_from_seeds_vmap
from my_utils import LATENT_SHAPE
def eval_model(
    FLAGS,
    train_state,
    train_state_teacher,
    step,
    dataset,
    dataset_valid,
    shard_data,
    vae_encode,
    vae_decode,
    update,
    get_fid_activations,
    imagenet_labels,
    visualize_labels,
    fid_from_stats,
    truth_fid_stats,
):
    with jax.spmd_mode('allow_all'):
        global_device_count = jax.device_count()
        key = jax.random.PRNGKey(42 + jax.process_index())
        batch_images,_, batch_labels = next(dataset)
        valid_images, valid_labels = next(dataset_valid)
        if FLAGS.model.use_stable_vae and 'latent' not in FLAGS.dataset_name:
            batch_images = vae_encode(key, batch_images)
            valid_images = vae_encode(key, valid_images)
        if 'latent' in FLAGS.dataset_name:
            # eps_valid = valid_images[..., :valid_images.shape[-1]//2]
            # batch_images = batch_images[..., batch_images.shape[-1]//2:]
            # valid_images = valid_images[..., valid_images.shape[-1]//2:]
            eps_valid = valid_images
            # This could be incorrect
        batch_labels_sharded, valid_labels_sharded = shard_data(batch_labels, valid_labels)
        labels_uncond = shard_data(jnp.ones(batch_labels.shape, dtype=jnp.int32) * FLAGS.model['num_classes']) # Null token
        eps = jax.random.normal(key, batch_images.shape)

        def process_img(img):
            if FLAGS.model.use_stable_vae:
                img = vae_decode(img[None])[0]
            img = img * 0.5 + 0.5
            img = jnp.clip(img, 0, 1)
            img = np.array(img)
            return img
        
        @partial(jax.jit, static_argnums=(5,))
        def call_model(train_state, images, t, dt, labels, use_ema=True):
            if use_ema and FLAGS.model.use_ema:
                call_fn = train_state.call_model_ema
            else:
                call_fn = train_state.call_model
            output = call_fn(images, t, dt, labels, train=False)
            return output

        print("Training Loss per T.")
        if FLAGS.model.denoise_timesteps == 128:
            fig, axs = plt.subplots(5, 8, figsize=(15, 12))
            d_list = [0, 1, 2, 3, 4, 5, 6, 7]
        else:
            fig, axs = plt.subplots(3, 6, figsize=(15, 8))
            d_list = [0, 1, 2, 3, 4, 5]
        for d in d_list:
            infos = None
            for t in np.arange(0, 32):
                t = t * (1.0 / 32)

                batch_images_n, batch_keys_n, batch_labels_n = next(dataset)
                if FLAGS.model.use_stable_vae and 'latent' not in FLAGS.dataset_name:
                    batch_images_n = vae_encode(key, batch_images_n)
                batch_noises_n=generate_tensor_from_seeds_vmap(batch_keys_n, LATENT_SHAPE)
                batch_images_sharded,batch_noises_sharded, batch_labels_sharded = shard_data(batch_images_n,batch_noises_n, batch_labels_n)
                
                _, info = update(train_state, train_state_teacher, batch_images_sharded,batch_noises_sharded, batch_labels_sharded, force_t=t, force_dt=d)
                info = jax.experimental.multihost_utils.process_allgather(info)
                if infos is None:
                    infos = jax.tree_map(lambda x: [x], info)
                else:
                    infos = jax.tree_map(lambda x, y: y + [x], info, infos)
            time_axis = np.arange(0, 32) / 32
            axs[0, d].plot(time_axis, infos['loss'])
            axs[0, d].set_title(f"All {d}")
            if FLAGS.model['train_type'] == 'shortcut':
                axs[1, d].plot(time_axis, infos['loss_flow'])
                axs[1, d].set_title(f"Flow {d}")
                axs[2, d].plot(time_axis, infos['loss_bootstrap'])
                axs[2, d].set_title(f"Bootstrap {d}")

            if jax.process_index() == 0:
                fig.tight_layout()
                wandb.log({f'mse': wandb.Image(fig)}, step=step)


        # print("One-step Denoising at various t.")
        # if 'latent' in FLAGS.dataset_name:
        #     eps = eps_valid
        # for dt_type in ['flow', 'shortcut']:
        #     if len(jax.local_devices()) == 8:
        #         if dt_type == 'flow':
        #             t = jnp.arange(8) / 8 # between 0 and 0.875
        #             t = jnp.tile(t, valid_images.shape[0] // 8) # [batch, etc]
        #             dt = 0
        #             dt_base = jnp.ones_like(t) * np.log2(FLAGS.model.denoise_timesteps)
        #         elif dt_type == 'shortcut':
        #             dt_base = jnp.array([0,0,0,1,2,3,4,5])
        #             if FLAGS.model.denoise_timesteps == 128:
        #                 dt_base = jnp.array([0,1,2,3,4,5,6,7])
        #             dt_base = jnp.tile(dt_base, valid_images.shape[0] // 8) # [batch, etc]
        #             dt = 2.0 ** (-dt_base)
        #             t = 1 - dt
        #         eps_tile = jnp.repeat(eps, 8, axis=0)[:valid_images.shape[0]]
        #         valid_images_tile = jnp.repeat(valid_images, 8, axis=0)[:valid_images.shape[0]]
        #         t_full = t[..., None, None, None]
        #         x_t = (1 - (1 - 1e-5) * t_full) * eps_tile + t_full * valid_images_tile
        #         x_t, t, dt_base = shard_data(x_t, t, dt_base)
        #         v_pred = call_model(train_state, x_t, t, dt_base, valid_labels_sharded if FLAGS.model.cfg_scale != 0 else labels_uncond)
        #         x_1_pred = x_t + v_pred * (1-t[..., None, None, None])
        #         x_t = jax.experimental.multihost_utils.process_allgather(x_t)
        #         x_1_pred = jax.experimental.multihost_utils.process_allgather(x_1_pred)
        #         valid_images_gather = jax.experimental.multihost_utils.process_allgather(shard_data(valid_images_tile))
        #         if jax.process_index() == 0:
        #             # valid_images_gather is [batchsize] wide. Every 8 corresponds to a timescale.
        #             fig, axs = plt.subplots(8, 4*3, figsize=(30, 30))
                    
        #             for j in range(min(4, valid_images_gather.shape[0] // 8)):
        #                 for k in range(8):
        #                     axs[k,3*j].imshow(process_img(valid_images_gather[j*8 + k]), vmin=0, vmax=1)
        #                     axs[k,3*j+1].imshow(process_img(x_t[j*8 + k]), vmin=0, vmax=1)
        #                     axs[k,3*j+2].imshow(process_img(x_1_pred[j*8 + k]), vmin=0, vmax=1)
        #             wandb.log({f'reconstruction_{dt_type}': wandb.Image(fig)}, step=step)
        #             plt.close(fig)

        print("Denoising at N steps")

        denoise_timesteps_list = [1, 2, 4, 8, 16, 32]
        if FLAGS.model.denoise_timesteps == 128:
            denoise_timesteps_list.append(128)
        if FLAGS.model.cfg_scale != 0:
            denoise_timesteps_list.append('cfg')
        for denoise_timesteps in denoise_timesteps_list:
            do_cfg = False
            if denoise_timesteps == 'cfg':
                denoise_timesteps = denoise_timesteps_list[-2]
                do_cfg = True
            all_x = []
            delta_t = 1.0 / denoise_timesteps
            x = eps # [local_batch, ...]
            x = shard_data(x) # [batch, ...] (on all devices)
            for ti in range(denoise_timesteps):
                t = ti / denoise_timesteps # From x_0 (noise) to x_1 (data)
                t_vector = jnp.full((eps.shape[0],), t)
                dt_base = jnp.ones_like(t_vector) * np.log2(denoise_timesteps)
                if FLAGS.model.train_type == 'livereflow' and denoise_timesteps < 128:
                    dt_base = jnp.zeros_like(t_vector)
                t_vector, dt_base = shard_data(t_vector, dt_base)
                if not do_cfg:
                    v = call_model(train_state, x, t_vector, dt_base, visualize_labels if FLAGS.model.cfg_scale != 0 else labels_uncond)
                else:
                    v_cond = call_model(train_state, x, t_vector, dt_base, visualize_labels)
                    v_uncond = call_model(train_state, x, t_vector, dt_base, labels_uncond)
                    v = v_uncond + FLAGS.model.cfg_scale * (v_cond - v_uncond)
                x = x + v * delta_t
                if denoise_timesteps <= 8 or ti % (denoise_timesteps // 8) == 0 or ti == FLAGS.model.denoise_timesteps-1:
                    np_x = jax.experimental.multihost_utils.process_allgather(x)
                    all_x.append(np.array(np_x))
            all_x = np.stack(all_x, axis=1) # [batch, timesteps, etc..]
            all_x = all_x[:, -8:]
            if jax.process_index() == 0:
                fig, axs = plt.subplots(8, 8, figsize=(30, 30))
                for j in range(8):
                    for t in range(min(8, all_x.shape[1])):
                        axs[t, j].imshow(process_img(all_x[j, t]), vmin=0, vmax=1)
                d_label = 'cfg' if do_cfg else denoise_timesteps
                wandb.log({f'sample_N/{d_label}': wandb.Image(fig)}, step=step)
                plt.close(fig)

        def do_fid_calc(cfg_scale, denoise_timesteps):
            activations = []
            images_shape = batch_images.shape
            num_generations = 4096
            print(f"Calc FID for CFG {cfg_scale} and denoise_timesteps {denoise_timesteps}")
            for fid_it in tqdm.tqdm(range(num_generations // FLAGS.batch_size)):
                key = jax.random.PRNGKey(42)
                key = jax.random.fold_in(key, fid_it)
                key = jax.random.fold_in(key, jax.process_index())
                eps_key, label_key = jax.random.split(key)
                x = jax.random.normal(eps_key, images_shape)
                labels = jax.random.randint(label_key, (images_shape[0],), 0, FLAGS.model.num_classes)
                x, labels = shard_data(x, labels)
                delta_t = 1.0 / denoise_timesteps
                for ti in range(denoise_timesteps):
                    t = ti / denoise_timesteps # From x_0 (noise) to x_1 (data)
                    t_vector = jnp.full((images_shape[0], ), t)
                    dt_base = jnp.ones_like(t_vector) * np.log2(denoise_timesteps)
                    if FLAGS.model.train_type == 'livereflow' and denoise_timesteps < 128:
                        dt_base = jnp.zeros_like(t_vector)
                    t_vector, dt_base = shard_data(t_vector, dt_base)
                    if cfg_scale == 1:
                        v = call_model(train_state, x, t_vector, dt_base, labels)
                    elif cfg_scale == 0:
                        v = call_model(train_state, x, t_vector, dt_base, labels_uncond)
                    else:
                        v_pred_uncond = call_model(train_state, x, t_vector, dt_base, labels_uncond)
                        v_pred_label = call_model(train_state, x, t_vector, dt_base, labels)
                        v = v_pred_uncond + cfg_scale * (v_pred_label - v_pred_uncond)
                    x = x + v * delta_t # Euler sampling.
                if FLAGS.model.use_stable_vae:
                    x = vae_decode(x) # Image is in [-1, 1] space.
                x = jax.image.resize(x, (x.shape[0], 299, 299, 3), method='bilinear', antialias=False)
                x = jnp.clip(x, -1, 1)
                acts = get_fid_activations(x)[..., 0, 0, :] # [devices, batch//devices, 2048]
                acts = jax.experimental.multihost_utils.process_allgather(acts)
                acts = np.array(acts)
                activations.append(acts)
            return activations
        
        if FLAGS.fid_stats is not None:
            denoise_timesteps_list = [1, 4, 32]
            if FLAGS.model.denoise_timesteps == 128:
                denoise_timesteps_list.append(128)
            if FLAGS.model.cfg_scale != 0:
                denoise_timesteps_list.append('cfg')
            for denoise_timesteps in denoise_timesteps_list:
                if denoise_timesteps == 'cfg':
                    activations = do_fid_calc(FLAGS.model.cfg_scale, FLAGS.model.denoise_timesteps)
                else:
                    activations = do_fid_calc(1 if FLAGS.model.cfg_scale != 0 else 0, denoise_timesteps)
                if jax.process_index() == 0:
                    activations = np.concatenate(activations, axis=0)
                    activations = activations.reshape((-1, activations.shape[-1]))
                    mu1 = np.mean(activations, axis=0)
                    sigma1 = np.cov(activations, rowvar=False)
                    fid = fid_from_stats(mu1, sigma1, truth_fid_stats['mu'], truth_fid_stats['sigma'])
                    print(f"FID for denoise_timesteps {denoise_timesteps} is {fid}")
                    wandb.log({f'fid/timesteps/{denoise_timesteps}': fid}, step=step)