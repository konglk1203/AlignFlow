from jax.tree_util import tree_map
from torch.utils.data import default_collate
import numpy as np

NUM_CLASSES_IMAGENET=1000
LATENT_SHAPE=(32,32,4)


import tensorflow as tf




class InfiniteDataLoader:
    """Wrapper for DataLoader to create an infinite, auto-shuffling data loader."""
    
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.data_iter = iter(self.dataloader)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        try:
            data = next(self.data_iter)
        except StopIteration:
            # Reached the end of the dataset, create a new iterator with reshuffled data
            self.data_iter = iter(self.dataloader)
            data = next(self.data_iter)
        return data
    
    def __len__(self):
        return len(self.dataloader)
    



def numpy_collate(batch):
    return tree_map(np.asarray, default_collate(batch))


