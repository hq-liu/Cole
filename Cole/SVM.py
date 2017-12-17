__author__='lhq'


from skimage.feature import local_binary_pattern
from skimage.feature import hog
from skimage.io import imread
from sklearn.externals import joblib
import data_input
import os
from PIL import Image
import numpy as np
from sklearn.svm import SVC
from sklearn import metrics
import time
from sklearn import tree


def get_hog_train(img_w,img_h):
    image_list, label_list = data_input.get_train_img()
    img_lists = np.empty((1, img_w* img_h))
    for image in image_list:
        img=image.reshape((img_w,img_h))
        fd,hog_img=hog(image=img,orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualise=True)
        hog_img=hog_img.reshape((1,img_w*img_h))
        img_lists = np.concatenate((img_lists, hog_img), axis=0)
    img_lists = np.delete(img_lists, 0, 0)
    label_list = [int(i) for i in label_list]
    label_list = np.array(label_list)

    return img_lists,label_list

def get_hog_test(img_w,img_h):

    image_list, label_list = data_input.get_test_img()
    img_lists = np.empty((1, img_w*img_h))
    for image in image_list:
        img = image.reshape((img_w, img_h))
        fd,hog_img = hog(image=img, orientations=8, pixels_per_cell=(16, 16),
                      cells_per_block=(1, 1), visualise=True)
        hog_img = hog_img.reshape((1, img_w* img_h))
        img_lists = np.concatenate((img_lists, hog_img), axis=0)
    img_lists = np.delete(img_lists, 0, 0)
    label_list = [int(i) for i in label_list]
    label_list = np.array(label_list)

    return img_lists, label_list

def get_lbp_train(img_w,img_h):

    image_list,label_list=data_input.get_train_img()
    img_lists = np.empty((1, img_w* img_h))
    for image_content in image_list:
        img=Image.open(image_content).convert('L')
        img=img.resize((img_w,img_h))
        img=np.array(img).reshape((img_w,img_h))
        lbp_img=local_binary_pattern(image=img,P=24,R=3)
        lbp_img=lbp_img.reshape((1,img_w*img_h))
        img_lists = np.concatenate((img_lists, lbp_img), axis=0)
    img_lists = np.delete(img_lists, 0, 0)
    label_list = [int(i) for i in label_list]
    label_list = np.array(label_list)

    return img_lists,label_list

def get_lbp_test(img_w,img_h):
    image_list, label_list = data_input.get_test_img()
    img_lists = np.empty((1, img_w*img_h))
    for image_content in image_list:
        img = Image.open(image_content).convert('L')
        img = img.resize((img_w, img_h))
        img = np.array(img).reshape((img_w, img_h))
        lbp_img = local_binary_pattern(image=img, P=24, R=3)
        lbp_img = lbp_img.reshape((1, img_w * img_h))
        img_lists = np.concatenate((img_lists, lbp_img), axis=0)
    img_lists = np.delete(img_lists, 0, 0)
    label_list = [int(i) for i in label_list]
    label_list = np.array(label_list)

    return img_lists, label_list

def classify_svm(img_lists,label_list):
    classifier=SVC(gamma=0.001,kernel='poly')
    classifier.fit(img_lists,label_list)
    # test_x,test_y=get_hog_test(64,64)
    test_x, test_y = data_input.get_test_img()
    test_x=test_x.reshape((150,4096))
    predicted=classifier.predict(test_x)
    for index,num in enumerate(predicted):
        print('%s:%s'%(index,num))
    print("Classification report for classifier %s:\n%s\n"
          % (classifier, metrics.classification_report(test_y, predicted)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(test_y, predicted))
    joblib.dump(classifier, "svc_model.m")

def classify_tree(img_lists,label_list):
    classifier=tree.DecisionTreeClassifier()
    classifier.fit(img_lists,label_list)
    start_time = time.time()
    test_x,test_y=get_hog_test(64,64)

    # test_x, test_y = data_input.get_test_img()
    # test_x=test_x.reshape((150,4096))

    predicted=classifier.predict(test_x)
    since=time.time()
    print(since-start_time)
    for index,num in enumerate(predicted):
        print('%s:%s'%(index,num))
    print("Classification report for classifier %s:\n%s\n"
          % (classifier, metrics.classification_report(test_y, predicted)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(test_y, predicted))
    joblib.dump(classifier, "tree_model.m")

img_list,label_list=get_hog_train(64,64)
# img_list,label_list=data_input.get_train_img()
# img_list=img_list.reshape((1050,4096))
# print(label_list)
classify_svm(img_list,label_list)
classify_tree(img_list,label_list)


