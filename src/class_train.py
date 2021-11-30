#%% Trainer Class Definition

#%% Import Modules

# Import Tensorflow
import tensorflow as tf

# Import Numpy
import numpy as np

# Import Utility
import datetime
import logging


#%% Trainer

class Trainer(object):
    
    def __init__(self, model, data_provider,
                 n_epoch, batch_size, learn_rate = 0.001,
                 dropout_rate = 0, l2_reg = 0, weight_option = False,
                 logdir = 'logs/', eval_freq = 1, restore = False):
    
        self.model = model
        self.data_provider = data_provider
        
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.learn_rate = learn_rate
        
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.weight_option = tf.constant(weight_option)
        
        self.logdir = logdir
        self.eval_freq = eval_freq
        
        self.restore = restore
        
        self.model.set_training_param(self.l2_reg, self.dropout_rate)
        
        self._build_loss()
        self._build_metric()
        self._build_optimiser()
        self._build_summary_writer()
        self._build_metric()
        
        self._build_checkpoint()       
        
        logging.info('Initialised Trainer.')
       
        
    def _build_loss(self):
        
        self.cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(
            reduction = tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
        
        logging.info('Loss Op build complete.')
        
        
    @tf.function
    def _loss_op(self, prediction, label, weight = None):
        
        with tf.name_scope('loss_op'):
            if self.weight_option:
                loss_entropy = self.cross_entropy(label, prediction,
                                                  sample_weight = weight)
            
            else:
                loss_entropy = self.cross_entropy(label, prediction)
                
            loss_reg = tf.math.reduce_sum(self.model.losses)
            loss = tf.add(loss_entropy, loss_reg)
            
        return loss


    def _build_optimiser(self):
        
        self.optimiser = tf.keras.optimizers.Adadelta(
            learning_rate = self.learn_rate)
        
        logging.info('Optimiser build complete.')
        
        
    @tf.function
    def _optimisation_op(self, image, label, weight = None):
                
        with tf.name_scope('optimisation_op'):
            with tf.GradientTape() as tape:
                pred = self.model(image, training = tf.constant(True))
                loss = self._loss_op(pred, label, weight)
    
            grad = tape.gradient(loss, self.model.trainable_variables)
            
            self.optimiser.apply_gradients(zip(grad,
                                               self.model.trainable_variables))
        
        return loss

        
    def _build_summary_writer(self):
    
        self.writer = dict()
        
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        
        logdir_step = '{0}//{1}'.format(self.logdir, stamp)
        self.writer['step'] = tf.summary.create_file_writer(logdir_step)
        
        logdir_train = '{0}//{1}//train'.format(self.logdir, stamp)
        self.writer['train'] = tf.summary.create_file_writer(logdir_train)
        
        logdir_test = '{0}//{1}//test'.format(self.logdir, stamp)
        self.writer['test'] = tf.summary.create_file_writer(logdir_test)
   
    
    def _build_metric(self):
        
        self.metric = dict()
        
        self.metric['loss'] = tf.keras.metrics.SparseCategoricalCrossentropy()
        self.metric['accuracy'] = tf.keras.metrics.SparseCategoricalAccuracy()
        
    
    def _metric_op(self, batch_type, epoch):
        
        n_data = len(self.data_provider.file_list[batch_type])
        n_step = n_data // self.batch_size
        n_rem = n_data % self.batch_size
        
        self.metric['loss'].reset_states()
        self.metric['accuracy'].reset_states()
        
        for i in range(n_step):
            
            image, label, weight = self.data_provider(self.batch_size,
                                                      batch_type)
            
            pred = self.model(image, training = tf.constant(False))
            
            if self.weight_option:
                self.metric['loss'].update_state(label, pred, weight)
                
            else:
                self.metric['loss'].update_state(label, pred)
                
            self.metric['accuracy'].update_state(label, pred)
            
            
        if n_rem != 0:
            image, label, weight = self.data_provider(n_rem,
                                                      batch_type)
            
            pred = self.model(image, training = tf.constant(False))
            
            if self.weight_option:
                self.metric['loss'].update_state(label, pred, weight)
                
            else:
                self.metric['loss'].update_state(label, pred)
                
            self.metric['accuracy'].update_state(label, pred)
        
        
        with self.writer[batch_type].as_default():
            loss = self.metric['loss'].result().numpy()
            tf.summary.scalar('Loss', loss, step = epoch)
            
            accuracy = self.metric['accuracy'].result().numpy()
            tf.summary.scalar('Accuracy', accuracy, step = epoch)
        
        self.writer[batch_type].flush()
        
        logging.info('{0} Batch | Loss = {1:.5} | Accuracy = {2:.3}'
                     .format(batch_type, loss, accuracy))

 
    def _build_checkpoint(self):
        
        self.ckpt = tf.train.Checkpoint(optimizer = self.optimiser,
                                        model = self.model)
        
        log_path = '{0}/checkpoint'.format(self.logdir)
        
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt,
                                                       log_path,
                                                       max_to_keep = 5)
        
        logging.info('Checkpoint Manager build complete.')
        
    
    def _checkpoint_op(self):
        
        if self.restore:
            
            latest_ckpt = self.ckpt_manager.latest_checkpoint
            
            self.ckpt.restore(latest_ckpt)
            
            logging.info('Load Training Checkpoint from {0}.'
                         .format(latest_ckpt))               
        
        
    def run_training(self):
        
        self._checkpoint_op()
        
        n_image_train = len(self.data_provider.file_list['train'])
        n_step_per_epoch = n_image_train // self.batch_size
        
        image, label, weight = self.data_provider(self.batch_size,
                                                          'train')
        tf.summary.trace_on(graph = True)
            
        loss = self._optimisation_op(image, label, weight)
        
        with self.writer['step'].as_default():
            tf.summary.trace_export(name = 'graph', step = 0)
        
        for epoch in range(self.n_epoch):
            for step in range(n_step_per_epoch):
                
                image, label, weight = self.data_provider(self.batch_size,
                                                          'train')
                
                loss = self._optimisation_op(image, label, weight)
                
                with self.writer['step'].as_default():
                    tf.summary.scalar('Loss', loss,
                                      step = epoch * n_step_per_epoch + step)
                    
                self.writer['step'].flush()
                
                logging.info('Epoch = {0}/{1} | Step = {2}/{3} | Loss = {4:.5}'
                             .format(epoch, self.n_epoch, step, n_step_per_epoch, loss))
                
            if ((epoch+1) % self.eval_freq) == 0:                
                self._metric_op('train', epoch)
                self._metric_op('test', epoch)
                
                self.ckpt_manager.save(checkpoint_number = epoch)
                
        self._metric_op('train', epoch + 1)
        self._metric_op('test', epoch + 1)        
        
        self.ckpt_manager.save(checkpoint_number = epoch + 1)