__author__='lhq'


import torch
from PIL import Image
import numpy as np
from torch.autograd import Variable
from sklearn.externals import joblib
from skimage.feature import hog

def nn_predict():
    classify=torch.load('model_conv.pth')
    classify.eval()
    print(classify)
    img=Image.open('4.JPG')

    width=img.size[0]
    height=img.size[1]
    if width%64 != 0:
        width -= width%64
    if height%64 != 0:
        height -= height%64
    sp_w=64
    sp_h=64

    label_list=[]
    density=0

    img_list = np.empty((1, 64,64,3))
    for w in range(1,width,sp_w):
        for h in range(1,height,sp_h):
            region = img.crop((w, h, w + sp_w, h + sp_h))
            img_x=np.array(region,dtype=np.float32).reshape((1,64,64,3))
            img_list=np.concatenate((img_list,img_x),axis=0)

    img_list=np.delete(img_list,0,0)
    img_list=np.array(img_list,dtype=np.float32)*1./255-0.5
    img_list=torch.from_numpy(img_list)


    img_x = Variable(img_list,volatile=True).cuda()
    out = classify(img_x)

    _, pred = torch.max(out, 1)
    print(pred)
    density =pred.sum()

    # density=sum(label_list)/len(label_list)
    print(density.data[0]/len(img_list))

def tree_predict():
    clf = joblib.load("tree_model.m")
    img = Image.open('4.JPG').convert('L')

    width = img.size[0]
    height = img.size[1]
    if width % 64 != 0:
        width -= width % 64
    if height % 64 != 0:
        height -= height % 64
    sp_w = 64
    sp_h = 64
    label_list = []
    density = 0
    img_list = np.empty((1, 4096))
    for w in range(1,width,sp_w):
        for h in range(1,height,sp_h):
            region = img.crop((w, h, w + sp_w, h + sp_h))
            fd, hog_img = hog(image=region, orientations=8, pixels_per_cell=(16, 16),
                              cells_per_block=(1, 1), visualise=True)
            hog_img = hog_img.reshape((1, 4096))
            img_x=np.array(hog_img,dtype=np.float32).reshape((1,4096))
            img_list=np.concatenate((img_list,img_x),axis=0)
    img_list = np.delete(img_list, 0, 0)
    predicted = clf.predict(img_list)
    pred=sum(predicted)
    density=pred/len(predicted)
    print(density)

def svc_predict():
    clf = joblib.load("svc_model.m")
    img = Image.open('4.JPG').convert('L')

    width = img.size[0]
    height = img.size[1]
    if width % 64 != 0:
        width -= width % 64
    if height % 64 != 0:
        height -= height % 64
    sp_w = 64
    sp_h = 64
    label_list = []
    density = 0
    img_list = np.empty((1, 4096))
    for w in range(1,width,sp_w):
        for h in range(1,height,sp_h):
            region = img.crop((w, h, w + sp_w, h + sp_h))
            img_x=np.array(region,dtype=np.float32).reshape((1,4096))
            img_list=np.concatenate((img_list,img_x),axis=0)
    img_list = np.delete(img_list, 0, 0)
    predicted = clf.predict(img_list)
    pred=sum(predicted)
    density=pred/len(predicted)
    print(density)

nn_predict()
# tree_predict()
# svc_predict()

