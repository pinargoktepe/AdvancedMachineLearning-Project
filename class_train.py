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
from functions import ourTrain
from Datasets import ClassesDataset

# Hyperparameters
model_save_name = 'clas'
model_load_name = 'first'
load = True
save = True
in_size = 227
tile_size = 56
batch_size = 32
n_epochs = 10
num_classes = 48
########

train_dataset = ClassesDataset("dataset2/train/",in_size=in_size, num_classes=num_classes)
val_dataset = ClassesDataset("dataset2/val",in_size=in_size, num_classes=num_classes)
test_dataset = ClassesDataset("dataset2/test",in_size=in_size, num_classes=num_classes)
print(len(train_dataset))
print(len(val_dataset))
print(len(test_dataset))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device: ' + ('gpu' if torch.cuda.is_available() else 'cpu'))

model = AlexNetSelf(tile_size,in_size,selftrain=False,num_classes=num_classes)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters())
loss_function = nn.CrossEntropyLoss()

train_losses, val_losses, train_acc, val_acc = ourTrain(model, train_loader, val_loader, optimizer, loss_function, device=device,
                     saveWeights=save, saving_path="models/model_"+model_save_name+".pth",
                     loadWeights=load,loading_path='models/model_'+model_load_name+'.pth',
                        print_every=100,n_epochs=n_epochs)

results = np.zeros((n_epochs,4))
results[:,0]=train_losses
results[:,1]=val_losses
results[:,2]=train_acc
results[:,3]=val_acc
np.savetxt('experiments/exp'+str(model_load_name)+'to'+str(model_save_name)+'.txt', results, fmt='%s')