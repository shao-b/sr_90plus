import paddle
import paddle.nn as nn

class SRCNN(nn.Layer):
    def __init__(self, num_channels=3, upscale_factor=4):
        super(SRCNN, self).__init__()
        self.upscale_factor = upscale_factor
        
        self.conv1 = nn.Conv2D(num_channels, 64, kernel_size=9, padding=4)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2D(64, 32, kernel_size=1, padding=0)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2D(32, num_channels, kernel_size=5, padding=2)
    
    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.upscale_factor, mode='bicubic', align_corners=False)
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.conv3(x)
        return x