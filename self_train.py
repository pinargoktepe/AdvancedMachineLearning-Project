import numpy as np
import os
import torch
import torch.nn as nn
from imageio import imread
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.transforms import RandomCrop, Resize, ToTensor, Normalize
import matplotlib.pyplot as plt
from alexnet import AlexNet, AlexNetSelf
from functions import ourTrain, plotCompare
from Datasets import SelfSupervisedDataset

# Hyperparameters
model_save_name = 'second'
model_load_name = 'first'
save = True
load = False
lr = 0.003
in_size = 227
tile_size = 56
batch_size = 32
n_epochs = 10
num_classes = 48
n_epochs = 30

k,l = divmod(in_size,tile_size)
if l != 0:
    print("Tile_size is not valid")

## Just to check how it works
dataset_f_test = SelfSupervisedDataset("dataset1/test",in_size=in_size,tile_size=tile_size,w_norm=False)
plotCompare(5, dataset_f_test)

train_dataset = SelfSupervisedDataset("dataset1/train/",in_size=in_size,tile_size=tile_size)
val_dataset = SelfSupervisedDataset("dataset1/val",in_size=in_size,tile_size=tile_size)
test_dataset = SelfSupervisedDataset("dataset1/test",in_size=in_size,tile_size=tile_size)
print(len(train_dataset))
print(len(val_dataset))
print(len(test_dataset))


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device: ' + ('gpu' if torch.cuda.is_available() else 'cpu'))

model = AlexNetSelf(tile_size,in_size,selftrain=True,num_classes=num_classes)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_function = nn.MSELoss()

train_losses, val_losses, train_acc, val_acc = ourTrain(model, train_loader, val_loader, optimizer, loss_function, device=device,
                            saveWeights=save,saving_path='models/model_'+model_save_name+'.pth',
                            loadWeights=load,loading_path='models/model_'+model_load_name+'.pth',
                            print_every=100, self_train=True, n_epochs=n_epochs)
results = np.zeros((n_epochs,2))
results[:,0]=train_losses
results[:,1]=val_losses
np.savetxt('experiments/exp'+str(model_load_name)+'to'+str(model_save_name)+'.txt', results, fmt='%s')
