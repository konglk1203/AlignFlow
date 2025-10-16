import jax
from tqdm import tqdm
from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec, NamedSharding

import tensorflow_datasets as tfds
import tensorflow as tf

from utils.stable_vae import StableVAE
import os
import numpy as np

from my_utils import NUM_CLASSES_IMAGENET

TEMP_DIR = os.environ["TEMP_DIR"]
NUM_CLASSES_IMAGENET = int(os.environ["NUM_CLASSES_IMAGENET"])
vae = StableVAE.create()
vae_encode = jax.jit(vae.encode_mean_std)


def save_vae(if_train):
    if if_train:
        vae_mean_path = os.path.join(TEMP_DIR, f"vae_mean.npy")
        vae_std_path = os.path.join(TEMP_DIR, f"vae_std.npy")
        label_path = os.path.join(TEMP_DIR, f"label.npy")

    else:
        vae_mean_path = os.path.join(TEMP_DIR, f"vae_mean_eval.npy")
        vae_std_path = os.path.join(TEMP_DIR, f"vae_std_eval.npy")
        label_path = os.path.join(TEMP_DIR, f"label_eval.npy")
    if (
        os.path.exists(vae_mean_path)
        and os.path.exists(vae_std_path)
        and os.path.exists(label_path)
    ):
        return
    os.makedirs(os.path.dirname(vae_mean_path), exist_ok=True)

    device_mesh = mesh_utils.create_device_mesh((jax.device_count(),))
    mesh = Mesh(devices=device_mesh, axis_names=("devices"))
    data_sharding = NamedSharding(mesh, PartitionSpec("devices"))
    vae_mean_list = []
    vae_std_list = []
    label_list = []
    flip_list = [False, True] if if_train else [False]
    for flip in flip_list:
        train_ds = tfds.load(
            "imagenet2012", split="train" if if_train else "validation"
        )

        def deserialization_fn(data):
            image = data["image"]
            min_side = tf.minimum(tf.shape(image)[0], tf.shape(image)[1])
            image = tf.image.resize_with_crop_or_pad(image, min_side, min_side)
            image = tf.image.resize(image, (256, 256), antialias=True)
            if flip:
                image = tf.image.flip_left_right(image)
            image = tf.cast(image, tf.float32) / 255.0
            image = (image - 0.5) / 0.5  # Normalize to [-1, 1]
            return image, data["label"]

        train_ds = train_ds.map(deserialization_fn)
        train_ds = train_ds.batch(128)
        train_ds = tfds.as_numpy(train_ds)

        if if_train:
            desc = (
                "processing training dataset (with flip)"
                if flip
                else "processing training dataset (without flip)"
            )
        else:
            desc = "processing validation dataset"

        for batch_images, batch_labels in tqdm(train_ds, desc=desc):
            if len(batch_images) % jax.device_count() == 0:
                batch_images = jax.device_put(batch_images, data_sharding)
            vae_mean, vae_std = vae_encode(batch_images)
            vae_mean_list += [vae_mean]
            vae_std_list += [vae_std]
            label_list += [batch_labels]

    vae_mean_tensor = np.concatenate(vae_mean_list)
    vae_std_tensor = np.concatenate(vae_std_list)
    label_tensor = np.concatenate(label_list)

    np.save(vae_mean_path, vae_mean_tensor)
    np.save(vae_std_path, vae_std_tensor)
    np.save(label_path, label_tensor)


def generate_index_list():
    index_list_path = os.path.join(TEMP_DIR, f"index_list.npy")

    if os.path.exists(index_list_path):
        raise Exception("Already exists!")
    label_path = os.path.join(TEMP_DIR, f"label.npy")
    label_list = np.load(label_path)
    index_list = [[] for _ in range(NUM_CLASSES_IMAGENET)]  # Temporary storage for

    for i, label in enumerate(tqdm(label_list)):
        index_list[label] += [i]
    np.save(index_list_path, np.array(index_list, dtype=object), allow_pickle=True)


if __name__ == "__main__":
    save_vae(if_train=True)
    save_vae(if_train=False)
    generate_index_list()
