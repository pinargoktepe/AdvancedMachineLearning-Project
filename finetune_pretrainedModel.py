import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from Datasets import ClassesDataset
from torchvision import models
from functions import ourTrain

# Hyperparameters
model_save_name = 'clas'
model_load_name = 'first'
load = False
save = True
lr=0.001
in_size = 227
batch_size = 32
n_epochs = 10
num_classes = 48
folder_name = "../DeepFashion/"
#Dataset for target task
train_dataset = ClassesDataset(folder_name+"dataset2/train/",in_size=in_size, num_classes=num_classes)
val_dataset = ClassesDataset(folder_name+"dataset2/val",in_size=in_size, num_classes=num_classes)
test_dataset = ClassesDataset(folder_name+"dataset2/test",in_size=in_size, num_classes=num_classes)
print(len(train_dataset))
print(len(val_dataset))
print(len(test_dataset))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device: ' + ('gpu' if torch.cuda.is_available() else 'cpu'))

#model
model = models.alexnet(pretrained=True)
#change the number of classes in the last layer
#num_ftrs = model.fc.in_features
#model.fc = nn.Linear(num_ftrs, num_classes)
num_ftrs = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_ftrs,num_classes)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device: ' + ('gpu' if torch.cuda.is_available() else 'cpu'))
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(),lr=lr)
loss_function = nn.CrossEntropyLoss()

train_losses, val_losses, train_acc, val_acc = ourTrain(model, train_loader, val_loader, optimizer, loss_function, device=device,
                     saveWeights=save, saving_path=folder_name+"models/model_"+model_save_name,
                     loadWeights=load,loading_path=folder_name+'models/model_'+model_load_name,
                        print_every=100,n_epochs=n_epochs)

results = np.zeros((n_epochs,4))
results[:,0]=train_losses
results[:,1]=val_losses
results[:,2]=train_acc
results[:,3]=val_acc
np.savetxt(folder_name+'experiments/exp_'+str(model_load_name)+'_to_'
           +str(model_save_name)+'_lr='+str(lr)+'_nclasses='+str(num_classes)+'_nepochs='+str(n_epochs)+'.txt', results, fmt='%s')
