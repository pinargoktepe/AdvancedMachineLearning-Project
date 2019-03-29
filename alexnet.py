import torch
import numpy as np
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AlexNet(nn.Module):
    def __init__(self, output_size):
        super(AlexNet, self).__init__()
        self.main = nn.Sequential(nn.Conv2d(3*227*227, 96*55*55, kernel_size=11*11, stride=4),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=3*3, stride=2),
                                  nn.Conv2d(96*27*27, 256*27*27, kernel_size=5*5, stride=2),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=3*3, stride=2),
                                  nn.Conv2d(256*27*27, 256*13*13, kernel_size=3*3, stride=1),
                                  nn.ReLU(),
                                  nn.Conv2d(256*13*13, 384*13*13, kernel_size=3*3, stride=1),
                                  nn.ReLU(),
                                  nn.Conv2d(384*13*13, 256*13*13, kernel_size=3*3, stride=1),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=3*3, stride=2),
                                  nn.Dropout(),
                                  nn.Linear(256*6*6, 4096),
                                  nn.ReLU(),
                                  nn.Dropout(),
                                  nn.Linear(4096, 4096),
                                  nn.ReLU(),
                                  nn.Linear(4096, output_size)
                                  )

    def forward(self, x):
        out = self.main(x)
        return out