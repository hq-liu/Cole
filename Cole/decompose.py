__author__='lhq'

import data_input
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

train_list,train_label=data_input.get_train_img()
test_list,test_label=data_input.get_test_img()
img=np.concatenate((train_list, test_list), axis=0)
label=np.concatenate((train_label,test_label),axis=0)


X_tsne=TSNE(learning_rate=100).fit_transform(test_list.reshape(84,1024))

plt.figure()
plt.subplot(111)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1],c=test_label)
plt.show()

