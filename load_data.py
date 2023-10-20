# Build dataGenerator for prediction

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# from keras.utils import Sequence
from tensorflow.keras.utils import Sequence
import time

def default_collate_fn(samples):
    X = np.array([sample[0] for sample in samples])
    Y = np.array([sample[1] for sample in samples])

    return X, Y

class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=8, dim=(224,224), n_channels=3,
                 n_classes=10, shuffle=False,num_workers=1,collate_fn=default_collate_fn):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        print(index)
        if index < self.__len__()-1:
            indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

            # Find list of IDs
            list_IDs_temp = [self.list_IDs[k] for k in indexes]
            # print(list_IDs_temp)
            samples = []
            
            if self.num_workers == 0:
                # Generate data
                X, y = self.__data_generation(list_IDs_temp)
            else:
                with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                    for sample in executor.map(self.__laod_data, list_IDs_temp):
                        samples.append(sample)

                X, y = self.collate_fn(samples)
        
        elif index == self.__len__()-1:
            list_IDs_temp = self.list_IDs[index*self.batch_size:]
            samples = []
            
            if self.num_workers == 0:
                # Generate data
                X, y = self.__data_generation(list_IDs_temp)
            else:
                with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                    for sample in executor.map(self.__laod_data, list_IDs_temp):
                        samples.append(sample)

                X, y = self.collate_fn(samples)
            


        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def __laod_data(self, ID):
        tmp = np.load(ID)[0]
        tmp_img = Image.fromarray(tmp)
        tmp_img = tmp_img.resize(self.dim)

        X = np.array(tmp_img)
        # print(ID)
        # X = np.array([1])
        # time.sleep(1)
        y = ID

        return (X,y)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        # y = np.empty((self.batch_size), dtype=int)
        # X = [i for i in range(self.batch_size)]
        y = [i for i in range(self.batch_size)]

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            # X[i,] = np.load(ID)
            # print(ID)
            tmp = np.load(ID)[0]
            tmp_img = Image.fromarray(tmp)
            tmp_img = tmp_img.resize(self.dim)

            X[i,] = np.array(tmp_img)


            # Store class
            # y[i] = self.labels[ID]
            y[i] = ID

        # return X, tf.keras.utils.to_categorical(y, num_classes=self.n_classes)
        return X, np.array(y)