import lmdb
from PIL import Image
import numpy as np
import os

import lmdb
from PIL import Image
import os
import pickle
from tqdm import tqdm

def convert_imagenet_to_lmdb(image_folder, lmdb_path, map_size=int(1e12)):
    """
    Converts an ImageNet-like folder structure to an LMDB database.

    Args:
        image_folder (str): Path to the root folder containing image subfolders (classes).
        lmdb_path (str): Path to save the LMDB database.
        map_size (int): Maximum size of the LMDB database in bytes.
    """
    env = lmdb.open(lmdb_path, map_size=map_size)

    idx = 0
    with env.begin(write=True) as txn:
        for class_name in tqdm(sorted(os.listdir(image_folder))):
            class_path = os.path.join(image_folder, class_name)
            if not os.path.isdir(class_path):
                continue

            for image_filename in os.listdir(class_path):
                if image_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.JPEG')):
                    image_path = os.path.join(class_path, image_filename)
                    try:
                        # Read image
                        #img = Image.open(image_path).convert('RGB')
                        #img_data = img.tobytes()  # Raw image bytes
                        #img_size = img.size  # (width, height)
                        #img_mode = img.mode  # 'RGB'

                        img_bin = None
                        with open(image_path, 'rb') as f:
                            img_bin = f.read() 
                        assert img_bin is not None

                        # Class label (from folder name)
                        label = int(class_name[1:])  # e.g., n01443537 → 1443537

                        # Create dictionary to store
                        sample = {
                            'image': img_bin,
                            'label': label,
                            #'image': img_data,
                            #'label': label,
                            #'img_size': img_size,
                            #'img_mode': img_mode,
                            #'filename': image_filename
                        }

                        # Serialize and write to LMDB
                        txn.put(f'{idx}'.encode(), pickle.dumps(sample))
                        idx += 1

                    except Exception as e:
                        print(f"Error processing {image_path}: {e}")

        # Write metadata
        txn.put(b'num_samples', str(idx).encode())
        print(f"Total {idx} samples written to LMDB.")

    env.close()
    print(f"ImageNet converted to LMDB at: {lmdb_path}")

# Example usage (replace with your actual paths)
image_folder_path = './train'
lmdb_output_path = '/home/ubuntu/code/MeanFlow/data/ImageNet/imagenet_train_lmdb'
convert_imagenet_to_lmdb(image_folder_path, lmdb_output_path)

image_folder_path = './val'
lmdb_output_path = '/home/ubuntu/code/MeanFlow/data/ImageNet/imagenet_val_lmdb'
convert_imagenet_to_lmdb(image_folder_path, lmdb_output_path)

