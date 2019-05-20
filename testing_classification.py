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
from alexnet import AlexNet, AlexNetSelf, AlexNetSelf_deep
from functions import ourTrain, classificationTest
from Datasets import ClassesDataset
from torch.optim.lr_scheduler import StepLR
folder_name = "/var/tmp/st9_data/"
# Hyperparameters
model_load_name = 'ss_deep_1_clsf_1'
load = True
save = False
lr=0.001
in_size = 227
tile_size = 75
batch_size = 64
n_epochs = 15
num_classes = 48
step_size = 5
gamma = 0.3
########

test_dataset = ClassesDataset(folder_name+"dataset2/test",in_size=in_size, num_classes=num_classes)
print("len test dataset: ", len(test_dataset))

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device: ' + ('gpu' if torch.cuda.is_available() else 'cpu'))

model = AlexNetSelf(tile_size,in_size,selftrain=False,num_classes=num_classes)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=0.001)
loss_function = nn.CrossEntropyLoss()
scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

test_accuracies_1, test_accuracies_3, test_accuracies_5 = classificationTest(model, test_loader, device=device,
                                                                             print_every=100, loading_path='models/model_'+model_load_name, folder_name=folder_name)

results = np.zeros((1,3))
results[:,0]=test_accuracies_1
results[:,1]=test_accuracies_3
results[:,2]=test_accuracies_5

np.savetxt(folder_name+'experiments/test_exp_'+str(model_load_name)+'_to_'
           +str(model_load_name)+'_lr='+str(lr)+'_step='+str(step_size)+'_gamma='+str(gamma)+'_nclasses='+str(num_classes)+'_nepochs='+str(n_epochs)+'.txt', results, fmt='%s')
