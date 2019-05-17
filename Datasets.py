import numpy as np
import os
from imageio import imread
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.transforms import RandomCrop, Resize, ToTensor, Normalize




class SelfSupervisedDataset(Dataset):

    def __init__(self, root_dir, in_size, tile_size, w_norm=True):
        self.root_dir = root_dir
        self.in_size = in_size
        self.tile_size = tile_size
        self.images = []
        self.w_norm = w_norm
        for r, d, f in os.walk(root_dir):
            for file in f:
                self.images.append(os.path.join(root_dir, file))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = imread(self.images[idx])
        w, h = image.shape[:2]
        if w<h:
            minim = w
        else:
            minim = h
        if self.w_norm:
            transf = transforms.Compose([
                RandomCrop(minim),
                Resize(self.in_size),
                ToTensor(),
                Normalize(mean=(0.5, 0.5, 0.5,), std=(0.5, 0.5, 0.5))
            ])
        else:
            transf = transforms.Compose([
                RandomCrop(minim),
                Resize(self.in_size),
                ToTensor()])

        image = Image.fromarray(image)
        img = transf(image)
        mask = np.random.rand(int(self.in_size/self.tile_size)**2)
        mask_s = np.sort(mask)
        new_img = img.clone()
        label = np.zeros(((int(self.in_size/self.tile_size)**2)*2), dtype=float)
        #label2 = np.zeros((int(self.in_size / self.tile_size) ** 2), dtype=float)
        for i,r in enumerate(mask):
            index = np.where(mask_s == r)
            index = int(index[0])
            k,l = divmod(i, int(self.in_size/self.tile_size))
            ko,lo = divmod(index, int(self.in_size/self.tile_size))
            new_img[:,(k*self.tile_size):(k*self.tile_size+self.tile_size),(l*self.tile_size):(l*self.tile_size+self.tile_size)] \
                = img[:,(ko * self.tile_size):(ko * self.tile_size + self.tile_size), (lo * self.tile_size):(lo * self.tile_size + self.tile_size)]
            #label1[i] = ko
            label[i*2] = ko
            label[i*2+1] = lo
            label = 2*label/(self.in_size/self.tile_size)-1
        return new_img, label


class ClassesDataset(Dataset):

    def __init__(self, root_dir, in_size, num_classes):
        self.root_dir = root_dir
        self.in_size = in_size
        self.images = []
        self.num_classes = num_classes
        for r, d, f in os.walk(root_dir):
            for dir in d:
                curr_dir = os.path.join(root_dir, dir)
                for r2, d2, f2 in os.walk(curr_dir):
                    for file in f2:
                        self.images.append(os.path.join(curr_dir, file))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = imread(self.images[idx])
        w, h = image.shape[:2]
        if w < h:
            minim = w
        else:
            minim = h
        transf = transforms.Compose([
            RandomCrop(minim),
            Resize(self.in_size),
            ToTensor(),
            Normalize(mean=(0.5, 0.5, 0.5,), std=(0.5, 0.5, 0.5))
        ])
        image = Image.fromarray(image)
        img = transf(image)
        label = os.path.basename(os.path.dirname(self.images[idx]))
        label = int(label) - 1
        labels = np.zeros((self.num_classes, 1))
        labels[label] = 1
        return img, label

