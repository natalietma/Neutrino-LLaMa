import torch
import torch.nn.functional as F
import torch.nn as nn
from torchinfo import summary


def my_relu6(x):
    return torch.clamp(F.relu(x), max=6)

def my_hard_swish(x):
    return x * my_relu6(x + 3.0) / 6.0

def my_return_activation(x, nl):
    if nl == 'HS':
        x = my_hard_swish(x)
    elif nl == 'RE':
        x = my_relu6(x)
    return x


class MySqueeze(nn.Module):
    def __init__(self, input_channels, ratio=16):
        super(MySqueeze, self).__init__()
        self.ratio = ratio
        self.input_channels = input_channels
        
        # Global Average Pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # (H, W) -> (1, 1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(input_channels, input_channels // ratio)
        self.fc2 = nn.Linear(input_channels // ratio, input_channels)
        
        # Hard sigmoid
        self.hard_sigmoid = nn.Sigmoid() 
    
    def forward(self, x):
        # Global Average Pooling
        y = self.avg_pool(x)
        y = y.view(y.size(0), -1)
        # Fully connected layers
        y = self.fc1(y)
        y = F.relu(y)
        y = self.fc2(y)
        y = self.hard_sigmoid(y).view(-1, self.input_channels, 1, 1)
        
        # Squeeze-and-Excitation
        out = x * y 
        return out


class MyConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, nl='RE'):
        super(MyConvBlock, self).__init__()
        # Define the convolutional layer
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size//2,  # Use 'same' padding
            bias=False
        )
        # Define the batch normalization layer
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        # Store the activation type
        self.nl = nl

    def forward(self, x):
        # Apply convolution
        x = self.conv(x)
        # Apply batch normalization
        x = self.bn(x)
        # Apply activation function
        x = my_return_activation(x, self.nl)
        return x

 
class MyBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, t, stride, squeeze, nl='RE'):
        super(MyBottleneck, self).__init__()
        
        self.expand_channels = in_channels * t
        self.stride = stride
        self.squeeze = squeeze
        self.nl = nl
        self.use_shortcut = stride==1 and in_channels==out_channels

        # Pointwise expension
        self.pointwise_expansion = MyConvBlock(in_channels, self.expand_channels)
        # Depthwise convolution
        self.depthwise_conv = nn.Conv2d(
            in_channels=self.expand_channels,
            out_channels=self.expand_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding='same',
            groups=self.expand_channels,
            bias=False
        )
        self.depthwise_bn = nn.BatchNorm2d(self.expand_channels)
        # Pointwise projection
        self.project_conv = nn.Conv2d(
            in_channels=self.expand_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding='same',
            bias=False
        )
        self.project_bn = nn.BatchNorm2d(out_channels)
        # Squeeze layer
        self.squeeze_layer = MySqueeze(out_channels) if squeeze else None
        # Shortcut layer
        if not self.use_shortcut:
            self.shortcut_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                padding='same',
                bias=False
            )
            self.shortcut_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # Pointwise expansion
        out = self.pointwise_expansion(x)
        # Depthwise convolution
        out = self.depthwise_conv(out)
        out = self.depthwise_bn(out)
        out = my_return_activation(out, self.nl)
        # Pointwise projection
        out = self.project_conv(out)
        out = self.project_bn(out)  
        # Apply Squeeze if enabled
        if self.squeeze_layer:
            out = self.squeeze_layer(out)
        # Shortcut connection
        if self.use_shortcut:
            out += x
        else:
            shortcut = self.shortcut_conv(x)
            shortcut = self.shortcut_bn(shortcut)
            out += shortcut
        return out


class MyInvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, t, n, squeeze, nl):
        super(MyInvertedResidualBlock, self).__init__()
        self.blocks = nn.ModuleList()
        # First bottleneck with stride
        self.blocks.append(MyBottleneck(in_channels, out_channels, kernel_size, t, stride, squeeze, nl))
        # Subsequent bottlenecks with stride=1
        for _ in range(1, n):
            self.blocks.append(MyBottleneck(out_channels, out_channels, kernel_size, t, 1, squeeze, nl))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class Subnet(nn.Module):
    def __init__(self):
        super(Subnet, self).__init__()
        self.my_conv_block = MyConvBlock(in_channels=1, out_channels=32, kernel_size=5, stride=2, nl='RE')
        self.my_inverted_residual_block = MyInvertedResidualBlock(
            in_channels=32, 
            out_channels=12, 
            kernel_size=3, 
            stride=1, 
            t=2, 
            n=1, 
            squeeze=False, 
            nl='RE'
        )
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        out = self.my_conv_block(x)
        out = self.my_inverted_residual_block(out)
        out = self.avg_pool(out)
        return out


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.subnet = Subnet()
        # Merge layers
        self.merge_pool1 = nn.AvgPool2d(kernel_size=2, padding=1)
        self.merge_block1 = MyInvertedResidualBlock(in_channels=24, out_channels=32, kernel_size=(3, 3), stride=1, t=6, n=3, squeeze=False, nl='RE')
        self.merge_pool2 = nn.AvgPool2d(kernel_size=2, padding=1)
        self.merge_block2 = MyInvertedResidualBlock(in_channels=32, out_channels=48, kernel_size=(3, 3), stride=1, t=6, n=4, squeeze=False, nl='RE')
        self.merge_block3 = MyInvertedResidualBlock(in_channels=48, out_channels=64, kernel_size=(3, 3), stride=1, t=6, n=3, squeeze=True, nl='HS')
        self.merge_pool3 = nn.AvgPool2d(kernel_size=2, padding=1)
        self.merge_block4 = MyInvertedResidualBlock(in_channels=64, out_channels=96, kernel_size=(3, 3), stride=1, t=6, n=3, squeeze=True, nl='HS')
        self.merge_block5 = MyInvertedResidualBlock(in_channels=96, out_channels=160, kernel_size=(3, 3), stride=1, t=6, n=1, squeeze=True, nl='HS')
        # Final layers
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(160, 256)
        self.dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(256, 3)

    def forward(self, input_x, input_y):
        x = self.subnet(input_x)
        y = self.subnet(input_y)
        # Following layer
        merge = torch.cat([x, y], dim=1)  # Maximum operation
        merge = self.merge_pool1(merge)
        merge = self.merge_block1(merge)
        merge = self.merge_pool2(merge)
        merge = self.merge_block2(merge)
        merge = self.merge_block3(merge)
        merge = self.merge_pool3(merge)
        merge = self.merge_block4(merge)
        merge = self.merge_block5(merge)

        merge = self.global_pool(merge)
        merge = torch.flatten(merge, 1)
        merge = self.fc(merge)
        merge = F.relu6(merge)  # Assuming 'HS' maps to ReLU6
        merge = self.dropout(merge)
        merge = self.fc2(merge)

        return merge


# Code below is for testing.
if __name__ == '__main__':
    model = MyModel()
    model.to('cuda:0')
    summary(model, input_size=[(1,1, 512, 512), (1,1, 512, 512)])