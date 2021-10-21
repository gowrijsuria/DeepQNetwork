import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0, pool=True):
        super(ConvBlock, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class DQNetwork(nn.Module):
    def __init__(self, input_channels, output_actions):
        super(DQNetwork, self).__init__()
        self.conv1 = ConvBlock(input_channels, 64, normalize=False)
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(128, 512)
        self.conv4 = ConvBlock(512, 512)
        self.conv5 = ConvBlock(512, 512)
        self.pool = nn.MaxPool2d(3)

        self.fc1 = nn.Linear(2049, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, output_actions)

    def forward(self, x, obstacle_area):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, obstacle_area], dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
        
if __name__=="__main__":

    x = torch.randn(2, 3, 100, 100)
    net = DQNetwork(3, 4)
    out = net(x)

    print(out.shape)