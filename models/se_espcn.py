import torch
import torch.nn as nn

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SE_ESPCN(nn.Module):
    def __init__(self, num_channels=3, upscale_factor=4):
        super(SE_ESPCN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=5, padding=2)
        self.se1 = SEBlock(64)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.se2 = SEBlock(64)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.conv3 = nn.Conv2d(128, 32, kernel_size=3, padding=1)
        self.se3 = SEBlock(32)
        self.relu3 = nn.ReLU(inplace=True)
        
        self.conv4 = nn.Conv2d(32, num_channels * (upscale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
    
    def forward(self, x):
        out1 = self.relu1(self.se1(self.conv1(x)))
        out2 = self.relu2(self.se2(self.conv2(out1)))
        concat = torch.cat([out1, out2], dim=1)
        out3 = self.relu3(self.se3(self.conv3(concat)))
        out = self.pixel_shuffle(self.conv4(out3))
        return out