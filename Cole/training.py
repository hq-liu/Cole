__author__='lhq'

import torch
from torch import nn
import os
from torchvision import transforms
from torch.utils.data import DataLoader
import data_input
import model
from torch import optim
from torch.autograd import Variable
from torchvision.utils import save_image
import time
import numpy as np
import time
from torchvision import models


batch_size=64
learning_rate=1e-3
n_epoch=50


img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
train_data=data_input.MyTrainDataset(transform=img_transform)
test_data=data_input.MyTestDataset(transform=img_transform)
trainloader=DataLoader(train_data,batch_size=batch_size,shuffle=True)
testloader=DataLoader(test_data,batch_size=batch_size,shuffle=True)

classifier_conv=model.conv_classify()


# classifier=model.logistic_regression()
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(classifier_conv.parameters(),lr=learning_rate,weight_decay=1e-6)
use_gpu=torch.cuda.is_available()
if use_gpu:
    classifier=classifier_conv.cuda()

for epoch in range(n_epoch):
    print('*' * 10)
    print('epoch {}'.format(epoch + 1))
    since = time.time()
    running_loss = 0.0
    running_acc = 0.0
    for i, data in enumerate(trainloader, 1):
        img, label = data
        # img = img.view(img.size(0), -1)  # 将图片展开成 64x64
        if use_gpu:
            img = Variable(img).cuda()
            label = Variable(label).cuda()
        else:
            img = Variable(img)
            label = Variable(label)
        # 向前传播
        out = classifier_conv(img)
        loss = criterion(out, label)
        running_loss += loss.data[0] * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        running_acc += num_correct.data[0]
        # 向后传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 300 == 0:
            print('[{}/{}] Loss: {:.6f}, Acc: {:.6f}'.format(
                epoch + 1, n_epoch, running_loss / (batch_size * i),
                running_acc / (batch_size * i)))
    print('Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}'.format(
        epoch + 1, running_loss / (len(train_data)), running_acc / (len(
            train_data))))
    classifier.eval()
    eval_loss = 0.
    eval_acc = 0.
    for data in testloader:
        start=time.time()
        img, label = data
        # img = img.view(img.size(0), -1)
        if use_gpu:
            img = Variable(img, volatile=True).cuda()
            label = Variable(label, volatile=True).cuda()
        else:
            img = Variable(img, volatile=True)
            label = Variable(label, volatile=True)
        out = classifier(img)
        stop=time.time()
        loss = criterion(out, label)
        eval_loss += loss.data[0] * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        eval_acc += num_correct.data[0]
    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
        test_data)), eval_acc / (len(test_data))))
    print('Time:{:.1f} s'.format(time.time() - since))
    print(stop-start)
    print()

# 保存模型
torch.save(classifier, './model_conv.pth')
