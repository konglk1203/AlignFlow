import jax
import jax.numpy as jnp
import numpy as np

def get_targets(FLAGS, key, train_state, images, noises, labels, force_t=-1, force_dt=-1):
    label_key, time_key, noise_key = jax.random.split(key, 3)
    info = {}

    # 1) =========== Sample dt. ============
    bootstrap_batchsize = FLAGS.batch_size // FLAGS.model['bootstrap_every']
    log2_sections = np.log2(FLAGS.model['denoise_timesteps']).astype(np.int32)
    if FLAGS.model['bootstrap_dt_bias'] == 0:
        dt_base = jnp.repeat(log2_sections - 1 - jnp.arange(log2_sections), bootstrap_batchsize // log2_sections)
        dt_base = jnp.concatenate([dt_base, jnp.zeros(bootstrap_batchsize-dt_base.shape[0],)])
        num_dt_cfg = bootstrap_batchsize // log2_sections
    else:
        dt_base = jnp.repeat(log2_sections - 1 - jnp.arange(log2_sections-2), (bootstrap_batchsize // 2) // log2_sections)
        dt_base = jnp.concatenate([dt_base, jnp.ones(bootstrap_batchsize // 4), jnp.zeros(bootstrap_batchsize // 4)])
        dt_base = jnp.concatenate([dt_base, jnp.zeros(bootstrap_batchsize-dt_base.shape[0],)])
        num_dt_cfg = (bootstrap_batchsize // 2) // log2_sections
    force_dt_vec = jnp.ones(bootstrap_batchsize, dtype=jnp.float32) * force_dt
    dt_base = jnp.where(force_dt_vec != -1, force_dt_vec, dt_base)
    dt = 1 / (2 ** (dt_base)) # [1, 1/2, 1/4, 1/8, 1/16, 1/32]
    dt_base_bootstrap = dt_base + 1
    dt_bootstrap = dt / 2

    # 2) =========== Sample t. ============
    dt_sections = jnp.power(2, dt_base) # [1, 2, 4, 8, 16, 32]
    t = jax.random.randint(time_key, (bootstrap_batchsize,), minval=0, maxval=dt_sections).astype(jnp.float32)
    t = t / dt_sections # Between 0 and 1.
    force_t_vec = jnp.ones(bootstrap_batchsize, dtype=jnp.float32) * force_t
    t = jnp.where(force_t_vec != -1, force_t_vec, t)
    t_full = t[:, None, None, None]

    # 3) =========== Generate Bootstrap Targets ============
    x_1 = images[:bootstrap_batchsize]
    # x_0 = jax.random.normal(noise_key, x_1.shape)
    x_0=noises[:bootstrap_batchsize] if noises!=None else jax.random.normal(noise_key, x_1.shape)
    x_t = (1 - (1 - 1e-5) * t_full) * x_0 + t_full * x_1
    bst_labels = labels[:bootstrap_batchsize]
    call_model_fn = train_state.call_model if FLAGS.model['bootstrap_ema'] == 0 else train_state.call_model_ema
    if not FLAGS.model['bootstrap_cfg']:
        v_b1 = call_model_fn(x_t, t, dt_base_bootstrap, bst_labels, train=False)
        t2 = t + dt_bootstrap
        x_t2 = x_t + dt_bootstrap[:, None, None, None] * v_b1
        x_t2 = jnp.clip(x_t2, -4, 4)
        v_b2 = call_model_fn(x_t2, t2, dt_base_bootstrap, bst_labels, train=False)
        v_target = (v_b1 + v_b2) / 2
    else:
        x_t_extra = jnp.concatenate([x_t, x_t[:num_dt_cfg]], axis=0)
        t_extra = jnp.concatenate([t, t[:num_dt_cfg]], axis=0)
        dt_base_extra = jnp.concatenate([dt_base_bootstrap, dt_base_bootstrap[:num_dt_cfg]], axis=0)
        labels_extra = jnp.concatenate([bst_labels, jnp.ones(num_dt_cfg, dtype=jnp.int32) * FLAGS.model['num_classes']], axis=0)
        v_b1_raw = call_model_fn(x_t_extra, t_extra, dt_base_extra, labels_extra, train=False)
        v_b_cond = v_b1_raw[:x_1.shape[0]]
        v_b_uncond = v_b1_raw[x_1.shape[0]:]
        v_cfg = v_b_uncond + FLAGS.model['cfg_scale'] * (v_b_cond[:num_dt_cfg] - v_b_uncond)
        v_b1 = jnp.concatenate([v_cfg, v_b_cond[num_dt_cfg:]], axis=0)

        t2 = t + dt_bootstrap
        x_t2 = x_t + dt_bootstrap[:, None, None, None] * v_b1
        x_t2 = jnp.clip(x_t2, -4, 4)
        x_t2_extra = jnp.concatenate([x_t2, x_t2[:num_dt_cfg]], axis=0)
        t2_extra = jnp.concatenate([t2, t2[:num_dt_cfg]], axis=0)
        v_b2_raw = call_model_fn(x_t2_extra, t2_extra, dt_base_extra, labels_extra, train=False)
        v_b2_cond = v_b2_raw[:x_1.shape[0]]
        v_b2_uncond = v_b2_raw[x_1.shape[0]:]
        v_b2_cfg = v_b2_uncond + FLAGS.model['cfg_scale'] * (v_b2_cond[:num_dt_cfg] - v_b2_uncond)
        v_b2 = jnp.concatenate([v_b2_cfg, v_b2_cond[num_dt_cfg:]], axis=0)
        v_target = (v_b1 + v_b2) / 2

    v_target = jnp.clip(v_target, -4, 4)
    bst_v = v_target
    bst_dt = dt_base
    bst_t = t
    bst_xt = x_t
    bst_l = bst_labels

    # 4) =========== Generate Flow-Matching Targets ============

    labels_dropout = jax.random.bernoulli(label_key, FLAGS.model['class_dropout_prob'], (labels.shape[0],))
    labels_dropped = jnp.where(labels_dropout, FLAGS.model['num_classes'], labels)
    info['dropped_ratio'] = jnp.mean(labels_dropped == FLAGS.model['num_classes'])

    # Sample t.
    t = jax.random.randint(time_key, (images.shape[0],), minval=0, maxval=FLAGS.model['denoise_timesteps']).astype(jnp.float32)
    t /= FLAGS.model['denoise_timesteps']
    force_t_vec = jnp.ones(images.shape[0], dtype=jnp.float32) * force_t
    t = jnp.where(force_t_vec != -1, force_t_vec, t)         # If force_t is not -1, then use force_t.
    t_full = t[:, None, None, None] # [batch, 1, 1, 1]

    # Sample flow pairs x_t, v_t.
    # x_0 = jax.random.normal(noise_key, images.shape)
    x_1 = images
    x_0=noises if noises!=None else jax.random.normal(noise_key, images.shape)
    x_t = x_t = (1 - (1 - 1e-5) * t_full) * x_0 + t_full * x_1
    v_t = v_t = x_1 - (1 - 1e-5) * x_0
    dt_flow = np.log2(FLAGS.model['denoise_timesteps']).astype(jnp.int32)
    dt_base = jnp.ones(images.shape[0], dtype=jnp.int32) * dt_flow

    # ==== 5) Merge Flow+Bootstrap ====
    bst_size = FLAGS.batch_size // FLAGS.model['bootstrap_every']
    bst_size_data = FLAGS.batch_size - bst_size
    x_t = jnp.concatenate([bst_xt, x_t[:bst_size_data]], axis=0)
    t = jnp.concatenate([bst_t, t[:bst_size_data]], axis=0)
    dt_base = jnp.concatenate([bst_dt, dt_base[:bst_size_data]], axis=0)
    v_t = jnp.concatenate([bst_v, v_t[:bst_size_data]], axis=0)
    labels_dropped = jnp.concatenate([bst_l, labels_dropped[:bst_size_data]], axis=0)
    info['bootstrap_ratio'] = jnp.mean(dt_base != dt_flow)

    info['v_magnitude_bootstrap'] = jnp.sqrt(jnp.mean(jnp.square(bst_v)))
    info['v_magnitude_b1'] = jnp.sqrt(jnp.mean(jnp.square(v_b1)))
    info['v_magnitude_b2'] = jnp.sqrt(jnp.mean(jnp.square(v_b2)))

    return x_t, v_t, t, dt_base, labels_dropped, info