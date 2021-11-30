### Layer Operations

#%% Import Library

import tensorflow as tf


#%% 2D Convolution

def conv2D(n_feature, kernel_size, stride, l2_reg):
    
    name = 'conv_{}x{}'.format(kernel_size[0],kernel_size[1])
    
    conv_op = tf.keras.layers.Conv2D(
        n_feature, kernel_size, stride,
        padding='same', data_format = 'channels_last',
        activation = None, use_bias = True,
        kernel_initializer = tf.keras.initializers.glorot_uniform(),
        bias_initializer = tf.keras.initializers.zeros(),
        kernel_regularizer = tf.keras.regularizers.l2(l2_reg),
        bias_regularizer = tf.keras.regularizers.l2(l2_reg),
        name = name)
    
    
    return conv_op


#%% Dense

def dense(unit, l2_reg):
    
    return tf.keras.layers.Dense(unit,
                                 activation = tf.keras.activations.softmax,
                                 kernel_initializer = tf.keras.initializers.glorot_uniform(),
                                 bias_initializer = tf.keras.initializers.zeros(),
                                 kernel_regularizer = tf.keras.regularizers.l2(l2_reg),
                                 bias_regularizer = tf.keras.regularizers.l2(l2_reg))


#%% Maxpool

def maxpool2D(pool_size, stride):
    
    name = 'maxpool_{}x{}'.format(pool_size[0], pool_size[1]) 
    
    pool_op = tf.keras.layers.MaxPool2D(pool_size, stride,
                                        padding = 'same',
                                        data_format = 'channels_last',
                                        name = name)
    
    return pool_op


#%% Avgpool

def global_avgpool2D():
       
    pool_op = tf.keras.layers.GlobalAveragePooling2D(                                               
                                               data_format = 'channels_last',
                                               name = "avgpool")
    
    return pool_op