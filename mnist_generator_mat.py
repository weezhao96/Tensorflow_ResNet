#%% Tensorboard Viewer

import tensorflow as tf

import numpy as np

import datetime

from scipy.io import savemat, loadmat


#%% Dataset

mnist = tf.keras.datasets.mnist

(data_train, label_train),(data_test, label_test) = mnist.load_data()
data_train, data_test = data_train / 255.0, data_test / 255.0

data_train = np.array(data_train, dtype = np.float32)
label_train = np.array(label_train, dtype = np.int32)

data_test = np.array(data_test, dtype = np.float32)
label_test = np.array(label_test, dtype = np.int32)


#%% Training Data

save_path_train = 'mnist_dataset//train'

buffer = dict()

for i in range(data_train.shape[0]):
    
    save_name = '{0}//data_{1:0>5}.mat'.format(save_path_train, i)
   
    buffer['image'] = data_train[i]
    buffer['label'] = label_train[i]
    
    savemat(save_name, buffer)
            

#%% Test Data
save_path_test = 'mnist_dataset//test'

buffer = dict()

for i in range(data_test.shape[0]):
    
    save_name = '{0}//data_{1:0>5}.mat'.format(save_path_test, i)
   
    buffer['image'] = data_test[i]
    buffer['label'] = label_test[i]
    
    savemat(save_name, buffer)