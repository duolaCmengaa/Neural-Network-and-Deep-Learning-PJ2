import torch
import torch.nn as nn
import torch.nn.functional as F

# Correct import
import sys
import os
PJ_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PJ_root)

from utils.mynn import init_weights_

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, dropout_prob=0.5):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion *
                          planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
        self.dropout = nn.Dropout(p=dropout_prob)  # Add Dropout Layer

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        out = self.dropout(out)  # Apply Dropout
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, dropout_prob=0.5, init_weights=True):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, dropout_prob=dropout_prob)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, dropout_prob=dropout_prob)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, dropout_prob=dropout_prob)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, dropout_prob=dropout_prob)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

        if init_weights:
            self._init_weights()

    def _make_layer(self, block, planes, num_blocks, stride, dropout_prob):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, dropout_prob=dropout_prob))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def _init_weights(self):
        for m in self.modules():
            init_weights_(m)

def ResNet18_with_dropout(dropout_prob=0.5):
    return ResNet(BasicBlock, [2, 2, 2, 2], dropout_prob=dropout_prob)
