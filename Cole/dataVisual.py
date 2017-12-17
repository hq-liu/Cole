from sklearn.manifold import TSNE
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt

cwd=os.getcwd()
data_dir1=cwd+'/dataset/train/mi/'
data_dir2=cwd+'/dataset/train/shu/'

img_lists = np.empty((1, 64, 64, 1))
label=[]
for file in os.listdir(data_dir1):
    img=Image.open(data_dir1+file).convert('L')
    img = np.array(img).reshape((1, 64, 64, 1))
    img_lists = np.concatenate((img_lists, img), axis=0)
    label.append(1)
for file in os.listdir(data_dir2):
    img=Image.open(data_dir2+file).convert('L')
    img = np.array(img).reshape((1, 64, 64, 1))
    img_lists = np.concatenate((img_lists, img), axis=0)
    label.append(0)

label=[int(i) for i in label]
label=np.array(label)
img_lists=np.delete(img_lists,0,0)

X_tsne=TSNE(learning_rate=100).fit_transform(img_lists.reshape(300,4096))
plt.figure()
plt.subplot(111)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1],c=label)
plt.show()
