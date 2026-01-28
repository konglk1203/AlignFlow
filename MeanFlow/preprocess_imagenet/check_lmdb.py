import lmdb
import pickle
import numpy as np

#env = lmdb.open('imagenet_train_lmdb')
env = lmdb.open('train_vae_latents_lmdb')
with env.begin() as txn:
    #print("num_samples:", txn.get(b'num_samples').decode())
    #first_sample = pickle.loads(txn.get(b'0'))
    #print("First sample label:", first_sample['label'])
    first_sample = pickle.loads(txn.get(b'0'))
    print("First sample:", first_sample['moments'])
    print("num_samples:", txn.get(b'num_samples').decode())
    print("First sample:", np.array(first_sample['moments']).shape)
    print("First sample:", np.array(first_sample['moments_flip']).shape)
    print("First sample:", np.array(first_sample['label']))
    print("First sample:", np.array(first_sample['filename']))
