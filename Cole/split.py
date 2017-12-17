from PIL import Image
import os

cwd=os.getcwd()
img_dir=cwd+'/pic/'
split_dir=cwd+'/split/'
if not os.path.exists(split_dir):
    os.mkdir(split_dir)

i=1
for file in os.listdir(img_dir):
    img=Image.open(img_dir+file)
    width=img.size[0]
    height=img.size[1]
    for w in range(1,width,64):
        for h in range(1,height,64):
            region = img.crop((w, h, w + 64, h + 64              ))
            region.save(split_dir + '/' + str(i) + '.jpg')
            i += 1

