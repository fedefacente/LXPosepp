#resnet 18
import torch.nn as nn
import torchvision.models as models
import torch
import numpy as np

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv1(x_cat))

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

class regressor_attention(nn.Module):
    def __init__(self):
        super(regressor_attention, self).__init__()

        # Load the pre-trained ResNet-18 model
        resnet = models.resnet18(weights=None, norm_layer=nn.InstanceNorm2d)
        # resnet = models.resnet18(weights = None, norm_layer=lambda num_features: nn.GroupNorm(8, num_features))
        # norm_layer=lambda num_features: nn.GroupNorm(8, num_features)
        # Modify the first layer to accept grayscale images
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1.weight = nn.Parameter(resnet.conv1.weight[:, 0:1, :, :])

        # Replace the remaining layers
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool

        # self.dropout = nn.Dropout(0.2)

        # Replace the last fully connected layer
        num_features = resnet.fc.in_features
        self.cbam = CBAM(512)
        self.num_features = resnet.fc.in_features
        self.fc1 = nn.Linear(num_features, 3)
        self.fc2 = nn.Linear(num_features, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.cbam(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)


        output1 = self.fc1(x)
        output2 = self.fc2(x)
        return output1, output2

    def extract_features(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.cbam(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x



class registration_attention(nn.Module):
    def __init__(self):
        super(registration_attention, self).__init__()

        # Load the pre-trained ResNet-18 model
        resnet = models.resnet18(weights=None, norm_layer=nn.InstanceNorm2d)

        self.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1.weight = nn.Parameter(resnet.conv1.weight[:, 0:2, :, :])

        # Replace the remaining layers
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        self.cbam = CBAM(512)

        # self.dropout = nn.Dropout(0.2)

        # Replace the last fully connected layer
        num_features = resnet.fc.in_features
        self.fc1 = nn.Linear(num_features, 3)
        self.fc2 = nn.Linear(num_features, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.cbam(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        output1 = self.fc1(x)
        output2 = self.fc2(x)
        return output1, output2

