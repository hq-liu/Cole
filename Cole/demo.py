from skimage import io,data,color
import matplotlib.pyplot as plt
import os

cwd=os.getcwd()

data_dir2=cwd+'/split/'
file_list2=os.listdir(data_dir2)
file_list2.sort(key=lambda x:int(x[:-4]))
num=1
for file in file_list2:
    img=io.imread(data_dir2+file)
    img_gray=color.rgb2gray(img)
    rows,cols=img_gray.shape
    for i in range(rows):
        for j in range(cols):
            if (img_gray[i,j]<=0.5):
                img_gray[i,j]=0
            else:
                img_gray[i,j]=1
    io.imsave(data_dir2+'grey_'+str(num)+'.jpg',img_gray)
    num += 1
# io.imshow(img_gray)

