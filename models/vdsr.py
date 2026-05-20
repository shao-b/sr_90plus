import torch
import torch.nn as nn

class VDSR(nn.Module):
    def __init__(self, num_channels=3, upscale_factor=4, num_layers=20):
        super(VDSR, self).__init__()
        self.upscale_factor = upscale_factor
        
        layers = []
        layers.append(nn.Conv2d(num_channels, 64, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(64, 64, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
        
        layers.append(nn.Conv2d(64, num_channels, kernel_size=3, padding=1))
        self.body = nn.Sequential(*layers)
    
    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.upscale_factor, mode='bicubic', align_corners=False)
        residual = x
        out = self.body(x)
        out += residual
        return out