
import torch.nn as nn
import torch.nn.functional as F

from resblock import ResBlock

class ResCNN(nn.Module):
    def __init__(self):
        super(ResCNN, self).__init__()
        # input shape [1, 28, 28]
        self.conv = nn.Conv2d(1, 16, [5,5], padding='same') # [16, 28, 28]
        self.res1 = ResBlock(16,16,5)
        self.pool = nn.MaxPool2d(2,2) # [16, 14, 14]
        self.res2 = ResBlock(16,32,3) # [32, 7, 7]
        self.res3 = ResBlock(32,64,3) # [64, 3, 3]
        self.gap = nn.AvgPool2d(kernel_size=(3,3))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv(x)
        x = self.res1(x)
        x = self.pool(x)
        x = self.res2(x)
        x = self.pool(x)
        x = self.res3(x)
        x = self.pool(x)
        x = self.gap(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

