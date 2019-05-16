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
from torch.optim.lr_scheduler import StepLR

folder_name = "/var/tmp/st9_data/"
# Hyperparameters
model_save_name = folder_name + 'second_v3'
model_load_name = folder_name + 'first'
save = True
load = False
lr = 0.0003
in_size = 227
tile_size = 75
batch_size = 32
num_classes=48
n_epochs = 30
##############

k,l = divmod(in_size,tile_size)
if l != 0:
    print("Warning: Tile_size is not valid")

## Just to check how it works
#dataset_f_test = SelfSupervisedDataset("dataset1/test",in_size=in_size,tile_size=tile_size,w_norm=False)
#plotCompare(5, dataset_f_test)

train_dataset = SelfSupervisedDataset(folder_name+"dataset1/train/",in_size=in_size,tile_size=tile_size)
val_dataset = SelfSupervisedDataset(folder_name+"dataset1/val",in_size=in_size,tile_size=tile_size)
test_dataset = SelfSupervisedDataset(folder_name+"dataset1/test",in_size=in_size,tile_size=tile_size)
print(len(train_dataset))
print(len(val_dataset))
print(len(test_dataset))


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device: ' + ('gpu' if torch.cuda.is_available() else 'cpu'))
device= torch.device('cuda')
model = AlexNetSelf(tile_size,in_size,selftrain=True,num_classes=num_classes)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_function = nn.MSELoss()
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

train_losses, val_losses, train_acc, val_acc = ourTrain(model, train_loader, val_loader, optimizer, loss_function, scheduler,
                                                        device=device, saveWeights=save, saving_path=folder_name+'models/model_'+model_save_name,
                                                        loadWeights=load,loading_path=folder_name+'models/model_'+model_load_name,print_every=100,
                                                        self_train=True, n_epochs=n_epochs)
results = np.zeros((n_epochs,2))
results[:,0]=train_losses
results[:,1]=val_losses
np.savetxt(folder_name+'experiments/exp_'+str(model_load_name)+'_to_'
           +str(model_save_name)+'_lr='+str(lr)+'_ntiles='+str(int(in_size/tile_size))+'_nepochs='+str(n_epochs)+'.txt', results, fmt='%s')

