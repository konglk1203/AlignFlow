import jax
import jax.numpy as jnp
import numpy as np

def get_targets(FLAGS, key, train_state, images, noises, labels, force_t=-1, force_dt=-1):
    time_key, noise_key = jax.random.split(key, 2)
    info = {}

    # 1) =========== Sample dt (based on current train step). ============
    dt_flow = np.log2(FLAGS.model['denoise_timesteps']).astype(jnp.int32)
    dt_base = jnp.floor(train_state.step / (FLAGS.max_steps / dt_flow))
    dt = 1 / (2 ** (dt_base))
    info['dt_base'] = jnp.mean(dt_base)

    # 2) =========== Sample t. ============
    dt_sections = jnp.power(2, dt_base) # [1, 2, 4, 8, 16, 32]
    t = jax.random.randint(time_key, (images.shape[0],), minval=0, maxval=dt_sections).astype(jnp.float32)
    t = t / dt_sections # Between 0 and 1.
    t_full = t[:, None, None, None]
    t2 = t + dt
    t2_full = t2[:, None, None, None]

    # 2) =========== Generate Bootstrap Targets ============
    x_1 = images
    x_0 = noises if noises!=None else jax.random.normal(noise_key, x_1.shape)
    x_t = (1 - (1 - 1e-5) * t_full) * x_0 + t_full * x_1
    x_t2 = (1 - (1 - 1e-5) * t2_full) * x_0 + t2_full * x_1

    v_b2 = train_state.call_model_ema(x_t2, t2, dt_base, labels, train=False)
    pred_x1 = x_t2 + (1 - t2_full) * v_b2
    v_target = (pred_x1 - x_t) / (1 - t[:, None, None, None])

    info['v_magnitude_bootstrap'] = jnp.sqrt(jnp.mean(jnp.square(v_target)))
    info['v_magnitude_b1'] = jnp.sqrt(jnp.mean(jnp.square(x_t2 - x_t)))
    info['v_magnitude_b2'] = jnp.sqrt(jnp.mean(jnp.square(v_b2)))

    return x_t, v_target, t, dt_base, labels, info