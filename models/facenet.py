import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Module


class InceptionBlock1a(Module):
    def __init__(self, in_channels):
        super(InceptionBlock1a, self).__init__()
        BN_EPSILON = 0.00001 
        
        self.branch_3x3_1x1 = nn.Conv2d(in_channels, 96, kernel_size=1, stride=1, padding=0, bias=False)
        self.branch_3x3_bn1 = nn.BatchNorm2d(96, eps=BN_EPSILON)
        self.branch_3x3_3x3 = nn.Conv2d(96, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.branch_3x3_bn2 = nn.BatchNorm2d(128, eps=BN_EPSILON)

        self.branch_5x5_1x1 = nn.Conv2d(in_channels, 16, kernel_size=1, stride=1, padding=0, bias=False)
        self.branch_5x5_bn1 = nn.BatchNorm2d(16, eps=BN_EPSILON)
        self.branch_5x5_5x5 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2, bias=False)
        self.branch_5x5_bn2 = nn.BatchNorm2d(32, eps=BN_EPSILON)
        
        self.branch_pool_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.branch_pool_1x1 = nn.Conv2d(in_channels, 32, kernel_size=1, stride=1, padding=0, bias=False)
        self.branch_pool_bn = nn.BatchNorm2d(32, eps=BN_EPSILON)

        self.branch_1x1_conv = nn.Conv2d(in_channels, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.branch_1x1_bn = nn.BatchNorm2d(64, eps=BN_EPSILON)


    def forward(self, x):
        x_3x3 = F.relu(self.branch_3x3_bn1(self.branch_3x3_1x1(x)))
        x_3x3 = F.relu(self.branch_3x3_bn2(self.branch_3x3_3x3(x_3x3)))
        
        x_5x5 = F.relu(self.branch_5x5_bn1(self.branch_5x5_1x1(x)))
        x_5x5 = F.relu(self.branch_5x5_bn2(self.branch_5x5_5x5(x_5x5)))
        
        x_pool = self.branch_pool_maxpool(x) # MaxPool with stride 2
        x_pool = F.relu(self.branch_pool_bn(self.branch_pool_1x1(x_pool)))
        x_pool = F.pad(x_pool, (3, 4, 3, 4), mode='constant', value=0)
        
        x_1x1 = F.relu(self.branch_1x1_bn(self.branch_1x1_conv(x)))
        
        inception = torch.cat([x_3x3, x_5x5, x_pool, x_1x1], dim=1)
        
        return inception


class InceptionBlock1b(nn.Module):
    def __init__(self, in_channels):
        super(InceptionBlock1b, self).__init__()
        BN_EPSILON = 0.00001 
        
        self.branch_3x3_1x1 = nn.Conv2d(in_channels, 96, kernel_size=1, stride=1, padding=0, bias=False)
        self.branch_3x3_bn1 = nn.BatchNorm2d(96, eps=BN_EPSILON)
        self.branch_3x3_3x3 = nn.Conv2d(96, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.branch_3x3_bn2 = nn.BatchNorm2d(128, eps=BN_EPSILON)

        self.branch_5x5_1x1 = nn.Conv2d(in_channels, 32, kernel_size=1, stride=1, padding=0, bias=False)
        self.branch_5x5_bn1 = nn.BatchNorm2d(32, eps=BN_EPSILON)
        self.branch_5x5_5x5 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2, bias=False)
        self.branch_5x5_bn2 = nn.BatchNorm2d(64, eps=BN_EPSILON)

        self.branch_pool_avgpool = nn.AvgPool2d(kernel_size=3, stride=3, padding=0)
        
        self.branch_pool_1x1 = nn.Conv2d(in_channels, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.branch_pool_bn = nn.BatchNorm2d(64, eps=BN_EPSILON)
        
        self.branch_1x1_conv = nn.Conv2d(in_channels, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.branch_1x1_bn = nn.BatchNorm2d(64, eps=BN_EPSILON)


    def forward(self, x):
        x_3x3 = F.relu(self.branch_3x3_bn1(self.branch_3x3_1x1(x)))
        x_3x3 = F.relu(self.branch_3x3_bn2(self.branch_3x3_3x3(x_3x3)))
        
        x_5x5 = F.relu(self.branch_5x5_bn1(self.branch_5x5_1x1(x)))
        x_5x5 = F.relu(self.branch_5x5_bn2(self.branch_5x5_5x5(x_5x5)))

        x_pool = self.branch_pool_avgpool(x)
        x_pool = F.relu(self.branch_pool_bn(self.branch_pool_1x1(x_pool)))
        
        x_pool = F.pad(x_pool, (4, 4, 4, 4), mode='constant', value=0)
        
        x_1x1 = F.relu(self.branch_1x1_bn(self.branch_1x1_conv(x)))

        inception = torch.cat([x_3x3, x_5x5, x_pool, x_1x1], dim=1)
        
        return inception