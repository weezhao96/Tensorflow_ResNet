### Data Provider Class

#%% Modules

# Standard Library
import glob
import logging
import random
import os

# Tensorflow
import tensorflow as tf

# Numpy
import numpy as np

# I/O Interface
from scipy.io import loadmat # MatDataProvider
import imageio # ImgDataProvider


#%% Base Data Provider

class BaseDataProvider(object):

    '''
    A base class which deals with the model I/O process. It contains the basic
    methods to loop through a data set for training ('train'), testing ('test')
    , and inference ('infer') operations.

    It is a base class which is not meant to be instantiated, and is inherited
    by different data provider classes which deals with the file I/O phase of
    input.

    Inherited Class Attribute Requirements:
        subClass.path - str:
            Contains the string of the main folder containing the data sets.
            The folder itself must contain subfolders named 'train'
            (for training), 'test' (for testing), or 'infer' (for inference),
            depending on the batch type of those datas. Only the subfolders
            themselves will contain the actual data sets required for the
            batch type operation.

        subClass.file_list - dict = {keys: file_list}
            keys (batch type): 'train', 'test' or 'infer'
            file_list : list of image path string in the 'key' subfolder

            Dictionary which have the keys based on the existing 3 batch type
            subfolders in the main folder.

        subClass.file_id - dict = {keys: file_id}
            keys (batch type): 'train', 'test' or 'infer'
            file_id : int, which iterates through the whole data set as the
                      the class is called.

        subClass.static_id - dict = {keys: static_id}
            keys (batch type): 'train', 'test' or 'infer'
            static_id : list - int, an integer id associated with every set of
                        data, which is shuffled together with the file_list.
                        Used for a fixed ID reference when naming image
                        prediction during evaluation.

    Inherited Class Method Requirements:
        subClass._file_read(batch_type) - method:
            A subClass unique method to read specific file types and returns them
            to the _next_set method.

            Outputs a set of multidimensional arrays, (ie: image, label, weight)
            with specific shape requirements.

            image shape: [n_row, n_col, n_channel]
            label shape: []
            weight shape: []

            If the arrays do not exist or apply for the specific batch type,
            it should return a None value.
    '''


    def __call__(self, batch_size, batch_type):

        '''
        Returns the image, label, and weight multidimensional arrays to feed
        into the network.

        Input:
            batch_size - int:
                The size of the batch.

            batch_type - str ('train', 'test', or 'infer'):
                Determines which folder the data will be read from.

        Output:
            image, label and weight arrays.
        '''


        pl_image, pl_label, pl_weight = self._next_set(batch_type)

        shape_image = np.shape(pl_image)

        image = np.zeros((batch_size, *shape_image), np.float32)
        label = np.zeros((batch_size), np.int32)
        weight = np.zeros((batch_size), np.float32)
        
        image[0] = pl_image
        label[0] = pl_label
        weight[0] = pl_weight

        for n in range(1, batch_size):
            image[n], label[n], weight[n] = self._next_set(batch_type)
        
        image = tf.constant(image)
        label = tf.constant(label)
        weight = tf.constant(weight)
           
        return image, label, weight


    def _next_set(self, batch_type):

        '''
        Returns a single batch of data to the call method.
        '''

        self._cycle_file_id(batch_type)
        image, label, weight = self._file_read(batch_type)
        
        return image, label, weight


    def _cycle_file_id(self, batch_type):

        '''
        Cycle through the whole list of data and iterates the file id every
        time _next_set is called. Shuffles the file and static id list and
        resets the file id when it finish iterating through the whole
        file list (ie: one complete epoch).
        '''

        self.file_id[batch_type] += 1

        if self.file_id[batch_type] >= len(self.file_list[batch_type]):
            self.file_id[batch_type] = 0

            id_list = list(zip(self.file_list[batch_type],
                               self.static_id[batch_type]))

            random.shuffle(id_list)

            self.file_list[batch_type], self.static_id[batch_type] = zip(*id_list)


#%% Matlab Data Provider

class MatDataProvider(BaseDataProvider):

    '''
    A BaseDataProvider, subClass which reads .mat files which contains, image,
    label, and weight.
    '''

    def __init__(self, path):

        '''
        Instantiate MatDataProvider.

        Input:
            path - str:
                Contains the path to the main dataset folder.
        '''
        
        self.path = path
        
        self._get_file_list()


    def _get_file_list(self):

        '''
        Seeks the subfolders ('train', 'test', 'infer') in path, and generates
        a file list, a current file id, and static id for each element in the
        file list for each subfolder.

        The file list, static id and current file id for all batch type are
        saved in respective dicts as {batch type : element}.

        Input:
            path - str:
                Contains the path to the main dataset folder.
        '''

        file_list = dict()
        file_id = dict()
        static_id = dict()

        for arg in ('train', 'test', 'infer'):
            file_list[arg] = glob.glob(self.path + '//' + arg + '//*.mat')
            file_id[arg] = -1 if len(file_list[arg]) > 0 else -np.inf
            static_id[arg] = [i for i in range(len(file_list[arg]))]

        self.file_list = file_list
        self.file_id = file_id
        self.static_id = static_id

        for arg in ('train', 'test', 'infer'):
            if file_id[arg] == -1:
                logging.info('Mat files in {0} folder detected.'.format(arg))


    def _file_read(self, batch_type):

        '''
        Reads a dataset from the batch_type folder. The data is selected from
        the file_list[ batch_type ][ file_id[batch_type] ].

        Input:
            batch_type - str ('train', 'test', or 'infer'):

        Output:
            image, label, weight - numpy arrays:
                Arrays with the shape required by the __call__ method in
                BaseDataProvider.
        '''

        file_list = self.file_list[batch_type]
        file_name = file_list[self.file_id[batch_type]]

        mat_dict = loadmat(file_name)

        image = mat_dict.pop('image', None)
        label = mat_dict.pop('label', -1)
        weight = mat_dict.pop('weight', None)

        image, label, weight = self._preprocess_data(image, label, weight)

        return image, label, weight


    @staticmethod
    def _preprocess_data(image, label, weight):

        '''
        Reshapes batch to the shape required by BaseDataProvider.
        '''

        if np.all(image != None):
            shape = np.shape(image)

            if len(shape) == 2:
                image = np.expand_dims(image, 2)

        return image, label, weight
    
    
#%% Image Data Provider

class ImageDataProvider(BaseDataProvider):
    
    def __init__(self, path, img_format):
        
        self.path = path
        self.img_format = img_format
        
        self._get_file_list()
        
        
    def _get_file_list(self):
        
        file_list = dict()
        file_id = dict()
        static_id = dict()
        
        for arg in ('train','test','infer'):
            path_format = '{0}//{1}//*.{2}'.format(self.path, arg,
                                                   self.img_format)
            file_list[arg] = glob.glob(path_format)
            file_id[arg] = -1 if len(file_list[arg]) > 0 else -np.inf
            static_id[arg] = [i for i in range(len(file_list[arg]))]
            
        self.file_list = file_list
        self.file_id = file_id
        self.static_id = static_id
        
        for arg in ('train', 'test', 'infer'):
            if file_id[arg] == -1:
                logging.info('Image files in {0} folder detected.'.format(arg))
                
        
    def _file_read(self, batch_type):
        
        file_list = self.file_list[batch_type]
        file_name_img = file_list[self.file_id[batch_type]]
        file_index = [s for s in file_name_img.split('.')[0].split('_') if s.isdigit()][0]
        
        file_name_label = 'label_{0}.txt'.format(file_index)
        file_name_weight = 'weight_{0}.txt'.format(file_index)
        
        image = imageio.imread(file_name_img)
        
        label_path = '{0}//{1}//{2}'.format(self.path, batch_type, file_name_label)
        if os.path.exists(label_path):
            with open(label_path, 'r') as file:
                label = int(file.readline())
                
        else:
            label = -1
        
        weight_path = '{0}//{1}//{2}'.format(self.path, batch_type, file_name_weight)
        if os.path.exists(weight_path):
            with open(weight_path, 'r') as file:
                weight = float(file.readline())
                
        else:
            weight = None        
            
        image, label, weight = self._preprocess_data(image, label, weight)
        
        return image, label, weight
    
    
    @staticmethod
    def _preprocess_data(image, label, weight):

        '''
        Reshapes batch to the shape required by BaseDataProvider, and convert
        image to float.
        '''
        
        image = image / 255.0
        
        
        if np.all(image != None):
            shape = np.shape(image)

            if len(shape) == 2:
                image = np.expand_dims(image, 2)

        return image, label, weight