import jax
import jax.numpy as jnp
import numpy as np

def get_targets(FLAGS, key, train_state, train_state_teacher, images, noises, labels, force_t=-1, force_dt=-1):
    time_key, noise_key = jax.random.split(key, 2)
    info = {}

    # 1) =========== Sample dt (based on current train step). ============
    dt_flow = np.log2(FLAGS.model['denoise_timesteps']).astype(jnp.int32)
    dt_base = dt_flow - jnp.floor(train_state.step / (FLAGS.max_steps / dt_flow)) - 1
    dt_base = jnp.ones(images.shape[0], dtype=jnp.int32) * dt_base
    dt = 1 / (2 ** (dt_base))
    dt_base_bootstrap = dt_base + 1
    dt_bootstrap = dt / 2
    info['dt_base'] = jnp.mean(dt_base)

    # 2) =========== Sample t. ============
    dt_sections = jnp.power(2, dt_base) # [1, 2, 4, 8, 16, 32]
    t = jax.random.randint(time_key, (images.shape[0],), minval=0, maxval=dt_sections).astype(jnp.float32)
    t = t / dt_sections # Between 0 and 1.
    t_full = t[:, None, None, None]

    # 3) =========== Generate Bootstrap Targets ============
    x_1 = images
    x_0 = noises if noises!=None else jax.random.normal(noise_key, x_1.shape)
    x_t = (1 - (1 - 1e-5) * t_full) * x_0 + t_full * x_1
    call_model_fn = train_state_teacher.call_model if FLAGS.model['bootstrap_ema'] == 0 else train_state_teacher.call_model_ema
    cfg_scale = jnp.where(dt_base == dt_flow-1, FLAGS.model['cfg_scale'], 1)[0]
    if not FLAGS.model['bootstrap_cfg']:
        v_b1 = call_model_fn(x_t, t, dt_base_bootstrap, labels, train=False)
        t2 = t + dt_bootstrap
        x_t2 = x_t + dt_bootstrap[:, None, None, None] * v_b1
        x_t2 = jnp.clip(x_t2, -4, 4)
        v_b2 = call_model_fn(x_t2, t2, dt_base_bootstrap, labels, train=False)
        v_target = (v_b1 + v_b2) / 2
    else:
        x_t_extra = jnp.concatenate([x_t, x_t], axis=0)
        t_extra = jnp.concatenate([t, t], axis=0)
        dt_base_extra = jnp.concatenate([dt_base_bootstrap, dt_base_bootstrap], axis=0)
        labels_extra = jnp.concatenate([labels, jnp.ones_like(labels, dtype=jnp.int32) * FLAGS.model['num_classes']], axis=0)
        v_b1_raw = call_model_fn(x_t_extra, t_extra, dt_base_extra, labels_extra, train=False)
        v_b_cond = v_b1_raw[:x_1.shape[0]]
        v_b_uncond = v_b1_raw[x_1.shape[0]:]
        v_b1 = v_b_uncond + cfg_scale * (v_b_cond - v_b_uncond)

        t2 = t + dt_bootstrap
        x_t2 = x_t + dt_bootstrap[:, None, None, None] * v_b1
        x_t2 = jnp.clip(x_t2, -4, 4)
        x_t2_extra = jnp.concatenate([x_t2, x_t2], axis=0)
        t2_extra = jnp.concatenate([t2, t2], axis=0)
        v_b2_raw = call_model_fn(x_t2_extra, t2_extra, dt_base_extra, labels_extra, train=False)
        v_b2_cond = v_b2_raw[:x_1.shape[0]]
        v_b2_uncond = v_b2_raw[x_1.shape[0]:]
        v_b2 = v_b2_uncond + cfg_scale * (v_b2_cond - v_b2_uncond)
        v_target = (v_b1 + v_b2) / 2

    info['v_magnitude_bootstrap'] = jnp.sqrt(jnp.mean(jnp.square(v_target)))
    info['v_magnitude_b1'] = jnp.sqrt(jnp.mean(jnp.square(v_b1)))
    info['v_magnitude_b2'] = jnp.sqrt(jnp.mean(jnp.square(v_b2)))

    return x_t, v_target, t, dt_base, labels, info