__author__='lhq'

import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

def get_train_img():
    """
    获取输入数据
    :return:
    """
    cwd=os.getcwd()
    mi_dir=cwd+'/dataset/train/mi/'
    shu_dir=cwd+'/dataset/train/shu/'
    file1=os.listdir(mi_dir)
    file0=os.listdir(shu_dir)
    # file1.sort(key=lambda x: int(x[:-4]))
    # file0.sort(key=lambda x: int(x[:-4]))
    mi_img_list=np.empty((1, 64, 64, 3))
    shu_img_list = np.empty((1, 64, 64, 3))
    label=[]
    for mi_img_content in file1:
        mi_img=Image.open(mi_dir+mi_img_content)
        mi_img = np.array(mi_img).reshape((1, 64, 64, 3))
        mi_img_list = np.concatenate((mi_img_list, mi_img), axis=0)
        label.append(1)
    for shu_img_content in file0:
        shu_img=Image.open(shu_dir+shu_img_content)
        shu_img = np.array(shu_img).reshape((1, 64, 64, 3))
        shu_img_list = np.concatenate((shu_img_list, shu_img), axis=0)
        label.append(0)
    mi_img_list = np.delete(mi_img_list, 0, 0)
    shu_img_list = np.delete(shu_img_list, 0, 0)
    label=[int(i) for i in label]
    label=np.array(label)
    img_list=np.concatenate((mi_img_list, shu_img_list), axis=0)
    return img_list,label

def get_test_img():
    """
    获取输入数据
    :return:
    """
    cwd=os.getcwd()
    mi_dir=cwd+'/dataset/test/mi/'
    shu_dir=cwd+'/dataset/test/shu/'
    file1=os.listdir(mi_dir)
    file0=os.listdir(shu_dir)
    # file1.sort(key=lambda x: int(x[:-4]))
    # file0.sort(key=lambda x: int(x[:-4]))
    mi_img_list=np.empty((1, 64, 64, 3))
    shu_img_list = np.empty((1, 64, 64, 3))
    label=[]
    for mi_img_content in file1:
        mi_img=Image.open(mi_dir+mi_img_content)
        mi_img = np.array(mi_img).reshape((1, 64, 64, 3))
        mi_img_list = np.concatenate((mi_img_list, mi_img), axis=0)
        label.append(1)
    for shu_img_content in file0:
        shu_img=Image.open(shu_dir+shu_img_content)
        shu_img = np.array(shu_img).reshape((1, 64, 64, 3))
        shu_img_list = np.concatenate((shu_img_list, shu_img), axis=0)
        label.append(0)
    mi_img_list = np.delete(mi_img_list, 0, 0)
    shu_img_list = np.delete(shu_img_list, 0, 0)
    label=[int(i) for i in label]
    label=np.array(label)
    img_list=np.concatenate((mi_img_list, shu_img_list), axis=0)
    return img_list,label

class MyTrainDataset(Dataset):
    def __init__(self,  transform=None, target_transform=None):
        imgs,label = get_train_img()
        self.imgs = imgs
        self.label=label
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img = self.imgs[index]
        label=self.label[index]
        if self.transform is not None:
            img = self.transform(img)
        return img,label

    def __len__(self):
        return len(self.imgs)

class MyTestDataset(Dataset):
    def __init__(self,  transform=None, target_transform=None):
        imgs,label = get_test_img()
        self.imgs = imgs
        self.label=label
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img = self.imgs[index]
        label=self.label[index]
        if self.transform is not None:
            img = self.transform(img)
        return img,label

    def __len__(self):
        return len(self.imgs)

