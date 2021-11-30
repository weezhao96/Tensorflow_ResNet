#%% Tensorboard Viewer

# Import Tensorflow
import tensorflow as tf

# Import Numpy
import numpy as np

# Import Source
from src.class_model import ResNet
from src.class_data_provider import MatDataProvider, ImageDataProvider
from src.class_train import Trainer

# Import Utility
import datetime
import logging


#%% Logging

logging.basicConfig(level = logging.INFO,
                    format = '%(process)d %(asctime)s %(message)s',
                    filename = '')


#%% Matlab Data Provider

#data_provider = MatDataProvider('mnist_dataset')
data_provider = ImageDataProvider('mnist_dataset_img','png')


#%% ResNet

model = ResNet(1, 10, 2, [2,1])

#%% Trainer

trainer = Trainer(model, data_provider, 1, 64,
                  learn_rate = 0.01,
                  dropout_rate = 0, l2_reg = 0.001,
                  restore = False, eval_freq = 1)

trainer.run_training()