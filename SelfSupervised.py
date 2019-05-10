import numpy as np
import os
from imageio import imread
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.transforms import RandomCrop, Resize, ToTensor, Normalize




class SelfSupervisedDataset(Dataset):

    def __init__(self, root_dir, in_size, tile_size):
        self.root_dir = root_dir
        self.in_size = in_size
        self.tile_size = tile_size
        self.images = []
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
        transf = transforms.Compose([
            RandomCrop(minim),
            Resize(self.in_size),
            ToTensor(),
            #Normalize(mean=(0.5, 0.5, 0.5,), std=(0.5, 0.5, 0.5))
        ])
        image = Image.fromarray(image)
        img = transf(image)
        mask = np.random.rand(int(self.in_size/self.tile_size)**2)
        mask_s = np.sort(mask)
        new_img = img.clone()
        label = np.zeros((int(self.in_size/self.tile_size)**2))
        for i,r in enumerate(mask):
            index = np.where(mask_s == r)
            index = int(index[0])
            k,l = divmod(i, int(self.in_size/self.tile_size))
            ko,lo = divmod(index, int(self.in_size/self.tile_size))
            new_img[:,(k*self.tile_size):(k*self.tile_size+self.tile_size),(l*self.tile_size):(l*self.tile_size+self.tile_size)] \
                = img[:,(ko * self.tile_size):(ko * self.tile_size + self.tile_size), (lo * self.tile_size):(lo * self.tile_size + self.tile_size)]
            label[i] = index
        return new_img, label


def plotCompare(image_idx, dataset):
    images=[]
    for r, d, f in os.walk("dataset1/train"):
        for file in f:
            images.append(os.path.join("dataset1/train", file))
    image = Image.open(images[image_idx])
    image.show()
    per_im, lab = dataset[image_idx]
    transforms.ToPILImage()(per_im).show()
    tile_size = int(np.sqrt(lab.shape[0]))
    lab_a = np.zeros((tile_size,tile_size))
    for i in range(tile_size):
        #k, l = divmod(i, tile_size)
        lab_a[i] = lab[i*tile_size:i*tile_size+tile_size]
    print(lab_a)
