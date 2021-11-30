### ResNet Class Definition

#%% Import Library

# Import Tensorflow
import tensorflow as tf

# Import Utility
import logging

# Import Source
from src.layer_op import conv2D, dense, global_avgpool2D, maxpool2D
from src.class_block import BottleNeckBlock


#%% Network Definition

class ResNet(tf.keras.Model):
    
    def __init__(self, input_channel, n_class,
                 n_super_block, n_block,
                 name = 'resnet'):
        
        super(ResNet, self).__init__(name = name, dtype = tf.float32)
         
        self.input_channel = input_channel
        self.n_class = n_class
        
        self.l2_reg = 0
        self.dropout_rate = 0
             
        self.n_super_block = n_super_block
        self.n_block = n_block
        self.block_dict = dict()
        
        for i_sup_block in range(self.n_super_block):
            
            dict_key = 'super_block_{0}'.format(i_sup_block)
            self.block_dict[dict_key] = []             
                
        
    def set_training_param(self, l2_reg, dropout_rate):
        
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
                
    
    def build(self, input_shape):
        
        filter_size = 64
        
        # Input Block        
        self.conv_in = conv2D(filter_size, (7,7), (2,2), self.l2_reg)
        self.bn_in = tf.keras.layers.BatchNormalization()
        self.maxpool_in = maxpool2D((3,3), (2,2))
        
        # Super Block
        filter_size = self._build_super_block(filter_size)
             
        # Dense Layer
        self.dense_avgpool = global_avgpool2D()
        self.dense_layer = dense(self.n_class, self.l2_reg)
              
        logging.info('ResNet build complete.')
        
        
    def _build_super_block(self, filter_size):
        
        # Super-Block
        for i_sup_block in range(self.n_super_block):
           
                dict_key = 'super_block_{}'.format(i_sup_block)
                
                if i_sup_block != 0:
                    filter_size *= 2
                           
                for i_block in range(self.n_block[i_sup_block]):
                    
                    if i_block == 0:
                        self.block_dict[dict_key].append(
                            BottleNeckBlock(filter_size,
                                            True,
                                            self.dropout_rate,
                                            self.l2_reg,
                                            name = dict_key))                        
                        
                    else:
                        self.block_dict[dict_key].append(
                            BottleNeckBlock(filter_size,
                                            False,
                                            self.dropout_rate,
                                            self.l2_reg,
                                            name = dict_key))
        
        logging.info('Superblock build complete.')
           
        return filter_size
    
    
    def call(self, x, training = False):
        
        # Input Layer
        with tf.name_scope('input_block'):
            x = self.conv_in(x)
            x = self.bn_in(x, training = training)
            x = self.maxpool_in(x)    
        
        # BottleNeckBlock
        for i_sup_block in range(self.n_super_block):
            
            block_name = 'super_block_{0}'.format(i_sup_block)
            
            block_array = self.block_dict[block_name]
            
            for i_block in range(self.n_block[i_sup_block]):
                
                x = block_array[i_block](x)      
                
        # Dense Layer
        with tf.name_scope('dense_layer'):
            x = self.dense_avgpool(x)
            x = self.dense_layer(x)
                    
        return x