#%% Inferencer Class Definition

# Import Tensorflow
import tensorflow as tf

import os

import logging

#%% Inferencer

class Inferencer(object):
    
    def __init__(self, model, data_provider, restore_path, output_path):
        
        self.model = model
        self.data_provider = data_provider
        
        self.restore_path = restore_path
        
        self.output_path = output_path
        if not os.path.exists(output_path):
            os.mkdir(output_path)
            
        self._build_checkpoint()
    

    def _build_checkpoint(self):
        
        self.ckpt = tf.train.Checkpoint(model = self.model)
        
        log_path = '{0}//checkpoint'.format(self.restore_path)
        
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt,
                                                       log_path,
                                                       max_to_keep = 5)
        
        logging.info('Checkpoint Manager build complete.')
        
    
    def _checkpoint_op(self):
               
        latest_ckpt = self.ckpt_manager.latest_checkpoint
        
        self.ckpt.restore(latest_ckpt)
        
        logging.info('Load Model Checkpoint from {0}.'
                     .format(latest_ckpt))                      
   
    
    def _output_result(self, pred_class):
        
        file_id = self.data_provider.file_id['infer']
        static_id = self.data_provider.static_id['infer'][file_id]
        file_name = '{0}//prediction_{1}.txt'.format(self.output_path,
                                                     static_id)
        
        with open(file_name, 'w') as file:
            file.write('{0}'.format(pred_class))
            
    
    def run_inference(self):
        
        self._checkpoint_op()
        
        n_data = len(self.data_provider.file_list['infer'])
        
        for n in range(n_data):
            image, _, _ = self.data_provider(1, 'infer')
            
            pred = self.model(image, training = tf.constant(False))
            
            pred_class = tf.math.argmax(pred,1).numpy()
            
            self._output_result(pred_class)
            
            logging.info('{0}/{1} image processed.'.format(n,n_data))