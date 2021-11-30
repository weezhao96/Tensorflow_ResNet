## Block Class Definition

#%% Import Library

# Import Tensorflow
import tensorflow as tf

# Import Layer Op
from src.layer_op import conv2D

# Import Utilities
import logging

#%% Layer

class BottleNeckBlock(tf.keras.layers.Layer):
    
    def __init__(self, filter_size, downsample_bool,
                 dropout_rate, l2_reg,
                 name = 'bottleneck_block'):
        
        super(BottleNeckBlock, self).__init__(name = name, dtype = tf.float32)
        
        self.filter_size = filter_size
        self.downsample_bool = downsample_bool
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        
        
    def build(self, input_shape):
        
        # Downsampling
        if self.downsample_bool:
            self.downsample = conv2D(self.filter_size * 4,
                                     (1,1), (2,2), self.l2_reg)
        else:
            self.downsample = tf.identity
            
                   
        # Conv In
        self.conv_in = conv2D(self.filter_size, 
                              (1,1), (1,1), self.l2_reg)
        self.bn_in = tf.keras.layers.BatchNormalization()
        self.relu_in = tf.keras.layers.ReLU()
        self.dropout_in = tf.keras.layers.Dropout(self.dropout_rate)      
           
        # Conv Mid
        self.conv_mid = conv2D(self.filter_size,
                               (3,3), (1,1), self.l2_reg)
        self.bn_mid = tf.keras.layers.BatchNormalization()
        self.relu_mid = tf.keras.layers.ReLU()
        self.dropout_mid = tf.keras.layers.Dropout(self.dropout_rate)
                  
        # Conv Out
        self.conv_out = conv2D(self.filter_size * 4,
                               (1,1), (1,1), self.l2_reg)
        self.bn_out = tf.keras.layers.BatchNormalization()
        self.relu_out = tf.keras.layers.ReLU()
        self.dropout_out = tf.keras.layers.Dropout(self.dropout_rate)
                   
        # Residual Op
        self.res_activation = tf.keras.layers.ReLU() 
        
        logging.info('BottleNeckBlock build complete.')
        
        
    def call(self, x, training = tf.constant(False)):          
        
        with tf.name_scope('downsample'):
            x = self.downsample(x)
        
        # Conv In
        with tf.name_scope('input_op'):
            conv = self.conv_in(x)
            conv = self.bn_in(conv, training = training)
            conv = self.relu_in(conv)
            if training:
                conv = self.dropout_in(conv, training = training)
        
        # Conv Mid
        with tf.name_scope('mid_op'):
            conv = self.conv_mid(conv)
            conv = self.bn_mid(conv, training = training)
            conv = self.relu_mid(conv)
            if training:
                conv = self.dropout_mid(conv, training = training)
        
        # Conv Out
        with tf.name_scope('output_op'):
            conv = self.conv_out(conv)
            conv = self.bn_out(conv, training = training)
            conv = self.relu_out(conv)
            if training:
                conv = self.dropout_out(conv, training = training) 
        
        # Residual Op
        with tf.name_scope('residual_op'):
            x = self.res_activation(tf.math.add(x, conv))  
                
        return x