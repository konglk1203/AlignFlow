# Imagenet SDOT dataset utils
Imagenet is a large dataset and we need to store seed rather than the whole noise matrix. See Rmk. 1 in the paper.

Environment variables:
`TEMP_DIR` for a temporary storage place to save generated dataset.
`MAX_NUM_EPOCH` for how many epoch needed. This number should be the largest possible number used for training and do not have to be accurate, e.g., if you need ~200 epoch, you can generate 500 epoch since generating dataset is super cheap in both computation and storage.

Please see MeanFlow/README.md or ShortcutModel/README.md for examples.