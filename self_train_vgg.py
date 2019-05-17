import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from vgg import vgg13_bn
from functions import ourTrain, plotCompare
from Datasets import SelfSupervisedDataset
from torch.optim.lr_scheduler import StepLR

folder_name = "../Desktop/DeepFashion/"
# Hyperparameters
model_save_name = folder_name + 'vgg_1'
model_load_name = folder_name + 'first'
save = True
load = False
lr = 0.0001
in_size = 224
tile_size = 74
batch_size = 64
num_classes=48
n_epochs = 30
##############

k,l = divmod(in_size,tile_size)
if l != 0:
    print("Warning: Tile_size is not valid")

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
model = vgg13_bn(tile_size,in_size,selftrain=True,progress=False,num_classes=num_classes)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)
loss_function = nn.MSELoss()
scheduler = StepLR(optimizer, step_size=10, gamma=0.3)

train_losses, val_losses, train_acc_1, train_acc_3, train_acc_5, val_acc_1, val_acc_3, val_acc_5 = ourTrain(model,
                                                        train_loader, val_loader, optimizer, loss_function, scheduler,
                                                        device=device, saveWeights=save, saving_path='models/model_'+model_save_name,
                                                        loadWeights=load,loading_path='models/model_'+model_load_name,print_every=100,
                                                        self_train=True, n_epochs=n_epochs, folder_name=folder_name)

results = np.zeros((n_epochs,2))
results[:,0]=train_losses
results[:,1]=val_losses
np.savetxt(folder_name+'experiments/exp_'+str(model_load_name)+'_to_'
           +str(model_save_name)+'_lr='+str(lr)+'_ntiles='+str(int(in_size/tile_size))+'_nepochs='+str(n_epochs)+'.txt', results, fmt='%s')

