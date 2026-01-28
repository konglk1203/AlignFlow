import os
import json

# imagenet class name mapping
class_names = sorted(os.listdir('/home/ubuntu/code/MeanFlow/data/ImageNet/train'))
class_to_idx = {name: idx for idx, name in enumerate(class_names)}
print(class_to_idx)

# Store the dictionary to a JSON file
with open('imagenet_class_to_idx.json', 'w') as f:
    json.dump(class_to_idx, f, indent=4)

