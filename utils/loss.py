import paddle
import paddle.nn as nn
from paddle.vision import models

class PerceptualLoss(nn.Layer):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg19(pretrained=True).features
        self.vgg_layers = vgg[:10].eval()
        for param in self.vgg_layers.parameters():
            param.stop_gradient = True
        self.mse = nn.MSELoss()
    
    def forward(self, x, y):
        x_features = self.vgg_layers(x)
        y_features = self.vgg_layers(y)
        return self.mse(x_features, y_features)

class MixedLoss(nn.Layer):
    def __init__(self, mse_weight=0.8, perceptual_weight=0.2):
        super(MixedLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.perceptual_loss = PerceptualLoss()
        self.mse_weight = mse_weight
        self.perceptual_weight = perceptual_weight
    
    def forward(self, x, y):
        mse = self.mse_loss(x, y)
        perceptual = self.perceptual_loss(x, y)
        return self.mse_weight * mse + self.perceptual_weight * perceptual