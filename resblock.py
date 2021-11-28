import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding='same')
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding='same')
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()

        self.shortcut_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding='same')
        self.shortcut_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        a = self.relu1(self.norm1(self.conv1(x)))
        x = self.relu2(self.shortcut_norm(self.shortcut_conv(x)) + self.norm2(self.conv2(a)))
        return x