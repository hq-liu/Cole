from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageEnhance
import os

cwd=os.getcwd()
data_dir1=cwd+'/dataset/train/mi/'
data_dir2=cwd+'/dataset/train/shu/'

# img=img_as_float(img)
i=1
file_list1=os.listdir(data_dir1)
file_list1.sort(key=lambda x:int(x[:-4]))
file_list2=os.listdir(data_dir2)
file_list2.sort(key=lambda x:int(x[:-4]))

for file in file_list1:

    image=Image.open(data_dir1+file)

    enh_con = ImageEnhance.Contrast(image)
    contrast = 0.5
    image_contrasted = enh_con.enhance(contrast)
    image_contrasted.save(data_dir1+'image_contrasted'+str(i)+'.jpg')

    enh_bri = ImageEnhance.Brightness(image)
    brightness = 1.2
    image_brightened = enh_bri.enhance(brightness)
    image_brightened.save(data_dir1+'image_brightened'+str(i)+'.jpg')

    enh_bri2 = ImageEnhance.Brightness(image)
    brightness2 = 0.6
    image_brightened2 = enh_bri2.enhance(brightness2)
    image_brightened2.save(data_dir1+'image_darked'+str(i)+'.jpg')

    i += 1
i=1
for file in file_list2:
    image = Image.open(data_dir2 + file)

    enh_con = ImageEnhance.Contrast(image)
    contrast = 0.5
    image_contrasted = enh_con.enhance(contrast)
    image_contrasted.save(data_dir2+'image_contrasted' + str(i) + '.jpg')

    enh_bri = ImageEnhance.Brightness(image)
    brightness = 1.2
    image_brightened = enh_bri.enhance(brightness)
    image_brightened.save(data_dir2+'image_brightened' + str(i) + '.jpg')

    enh_bri2 = ImageEnhance.Brightness(image)
    brightness2 = 0.6
    image_brightened2 = enh_bri2.enhance(brightness2)
    image_brightened2.save(data_dir2+'image_darked' + str(i) + '.jpg')

    i += 1
    # 对比度增强

    # 锐度增强
#     enh_sha = ImageEnhance.Sharpness(image_contrasted)
#     sharpness = 3.0
#     image_sharped = enh_sha.enhance(sharpness)
#     image_sharped.show()
# image_sharped.save('image_sharped.jpg')

# result=exposure.is_low_contrast(img)
# print(result)
#
# gam1= exposure.adjust_gamma(img, 2)   #调暗
# gam2= exposure.adjust_gamma(img, 0.5)  #调亮
#
# plt.subplot(131)
# plt.title('origin image')
# plt.imshow(img,plt.cm.gray)
# plt.axis('off')
#
# plt.subplot(132)
# plt.title('gamma=2')
# plt.imshow(gam1,plt.cm.gray)
# plt.axis('off')
#
# plt.subplot(133)
# plt.title('gamma=0.5')
# plt.imshow(gam2,plt.cm.gray)
# plt.axis('off')
#
# plt.show()