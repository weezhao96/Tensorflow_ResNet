#%% Tensorboard Viewer

# Import Tensorflow
import tensorflow as tf

# Import Numpy
import numpy as np

# Import Source
from src.class_model import ResNet
from src.class_data_provider import MatDataProvider
from src.class_inference import Inferencer

# Import Utility
import logging


#%% Logging

logging.basicConfig(level = logging.INFO,
                    format = '%(process)d %(asctime)s %(message)s',
                    filename = '')

#%% Matlab Data Provider

data_provider = MatDataProvider('mnist_dataset')

#%% ResNet

model = ResNet(1, 10, 2, [2,1])

#%% Inferencer

inferencer = Inferencer(model, data_provider, 'logs', 'output')

inferencer.run_inference()