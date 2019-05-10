import numpy as np
import os
from imageio import imread
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.transforms import RandomCrop, Resize, ToTensor, Normalize
import matplotlib.pyplot as plt
from SelfSupervised import SelfSupervisedDataset, plotCompare

# Hyperparameters
in_size = 228
tile_size = 57
batch_size = 32

k,l = divmod(in_size,tile_size)
if l != 0:
    print("Tile_size is not valid")


train_dataset = SelfSupervisedDataset("dataset1/train/",in_size=in_size,tile_size=tile_size)
val_dataset = SelfSupervisedDataset("dataset1/val",in_size=in_size,tile_size=tile_size)
test_dataset = SelfSupervisedDataset("dataset1/test",in_size=in_size,tile_size=tile_size)
print(len(train_dataset))
print(len(val_dataset))
print(len(test_dataset))

#plt.ion()

plotCompare(12, train_dataset)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=64)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=64)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=32)