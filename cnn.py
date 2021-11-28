
import torch.nn as nn
import torch.nn.functional as F

class CNNetwork(nn.Module):
    def __init__(self):
        super(CNNetwork, self).__init__()
        # input shape [1, 28, 28]
        self.conv1 = nn.Conv2d(1, 2, [5,5], padding='same') # [2, 28, 28]
        self.pool = nn.MaxPool2d(2,2) # [2, 14, 14]
        self.conv2 = nn.Conv2d(2, 4, [5,5], padding='same') # [4, 14, 14]
        self.conv3 = nn.Conv2d(4, 8, [3,3], padding='same') # [8, 7, 7]
        self.conv4 = nn.Conv2d(8, 16, [3,3], padding='same') # [16, 3, 3]
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16*3*3, 150)
        self.fc2 = nn.Linear(150, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

