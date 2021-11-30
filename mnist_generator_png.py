#%% Tensorboard Viewer

import tensorflow as tf

import numpy as np

import datetime

import imageio

#%% Dataset

mnist = tf.keras.datasets.mnist

(data_train, label_train),(data_test, label_test) = mnist.load_data()


#%% Training Data

save_path_train = 'mnist_dataset_img//train'

buffer = dict()

for i in range(1000):
    
    save_name = '{0}//data_{1:0>5}'.format(save_path_train, i)
    save_name_label = '{0}//label_{1:0>5}'.format(save_path_train, i)

    
    buffer['image'] = data_train[i]
    buffer['label'] = label_train[i]

    imageio.imwrite('{0}.{1}'.format(save_name, 'png'), buffer['image'])
    
    with open('{0}.{1}'.format(save_name_label, 'txt'), 'w') as file:
        file.write('{0}'.format(buffer['label']))
    
    

#%% Test Data
save_path_test = 'mnist_dataset_img//test'

buffer = dict()

for i in range(1000):
    
    save_name = '{0}//data_{1:0>5}'.format(save_path_test, i)
    save_name_label = '{0}//label_{1:0>5}'.format(save_path_test, i)

    
    buffer['image'] = data_test[i]
    buffer['label'] = label_test[i]

    imageio.imwrite('{0}.{1}'.format(save_name_label, 'png'), buffer['image'])
    
    with open('{0}.{1}'.format(save_name, 'txt'), 'w') as file:
        file.write('{0}'.format(buffer['label']))