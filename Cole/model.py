__author__='lhq'

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from torchvision import models


class logistic_regression(nn.Module):

    def __init__(self):
        super(logistic_regression, self).__init__()
        self.logistic=nn.Linear(4096,2)
    def forward(self, x):
        out=self.logistic(x)
        return out

class fc_classify(nn.Module):

    def __init__(self):
        super(fc_classify, self).__init__()
        self.fc_classify=nn.Sequential(
            nn.Linear(4096,128),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(128,2)
        )

    def forward(self, x):
        out=F.relu(self.fc_classify(x))
        return out

class conv_classify(nn.Module):

    def __init__(self,num_classes=2):
        super(conv_classify, self).__init__()
        self.conv1=nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3,stride=1,padding=2) #16*64*64
        self.pool1=nn.MaxPool2d(kernel_size=2) #16*32*32
        self.bn1=nn.BatchNorm2d(16)
        self.conv2=nn.Conv2d(in_channels=16,out_channels=32,kernel_size=5,stride=1,padding=2) #32*32*32
        self.pool2=nn.MaxPool2d(kernel_size=2) #32*16*16
        self.bn2=nn.BatchNorm2d(32)
        self.fc1=nn.Linear(in_features=32*16*16,out_features=512)
        self.bn3=nn.BatchNorm2d(512)
        self.out=nn.Linear(in_features=512,out_features=num_classes)

    def forward(self, x):
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.pool1(x)
        x=self.conv2(x)
        x=self.bn2(x)
        x=self.pool2(x)

        x=x.view(x.size(0), -1)
        x=F.relu(self.bn3(self.fc1(x)))
        x=self.out(x)
        return F.softmax(x)

