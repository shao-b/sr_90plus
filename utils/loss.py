import torch
import torch.nn as nn
from torchvision import models

class PerceptualLoss(nn.Module):
    def __init__(self, device):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg19(pretrained=True).features.to(device)
        self.vgg_layers = vgg[:10].eval()
        for param in self.vgg_layers.parameters():
            param.requires_grad = False
        self.mse = nn.MSELoss()
    
    def forward(self, x, y):
        x_features = self.vgg_layers(x)
        y_features = self.vgg_layers(y)
        return self.mse(x_features, y_features)

class MixedLoss(nn.Module):
    def __init__(self, device, mse_weight=0.8, perceptual_weight=0.2):
        super(MixedLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.perceptual_loss = PerceptualLoss(device)
        self.mse_weight = mse_weight
        self.perceptual_weight = perceptual_weight
    
    def forward(self, x, y):
        mse = self.mse_loss(x, y)
        perceptual = self.perceptual_loss(x, y)
        return self.mse_weight * mse + self.perceptual_weight * perceptual