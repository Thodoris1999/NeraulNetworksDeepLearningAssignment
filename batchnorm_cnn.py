
import torch.nn as nn
import torch.nn.functional as F

class BatchnormCNN(nn.Module):
    def __init__(self):
        super(BatchnormCNN, self).__init__()
        # input shape [1, 28, 28]
        self.conv1 = nn.Conv2d(1, 16, [5,5], padding='same') # [16, 28, 28]
        self.norm1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2,2) # [2, 14, 14]
        self.conv2 = nn.Conv2d(16, 16, [5,5], padding='same') # [16, 14, 14]
        self.norm2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, [3,3], padding='same') # [32, 7, 7]
        self.norm3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 64, [3,3], padding='same') # [64, 3, 3]
        self.norm4 = nn.BatchNorm2d(64)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64*3*3, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.norm1(x)
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.norm2(x)
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.norm3(x)
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = self.norm4(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

