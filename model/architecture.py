from .configurations import *

import os
import glob

import tensorflow as tf
import keras
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.random.set_seed(1)


import numpy as  np
import datetime
import pandas as pd
import shutil

import math

keras.saving.get_custom_objects().clear()

@keras.saving.register_keras_serializable()
class LeNetLike(keras.Model):
    def __init__(self, kernel_size, filters, pool_size, activation_type=None, dropout_rate=0.2,
                 **kwargs):
        super(LeNetLike, self).__init__(**kwargs)
        print(activation_type)
        self.kernel_size = kernel_size
        self.filters = filters
        self.pool_size = pool_size
        self.activation_type = activation_type
        self.dropout_rate = dropout_rate
        
        self.conv1 = keras.layers.Conv1D(filters=4 * filters,
                                         kernel_size=kernel_size, 
                                         strides=2,
                                         padding='same',
                                         activation=activation_type)
        
        self.conv2 = keras.layers.Conv1D(filters=4 * filters,
                                         kernel_size=kernel_size,
                                         padding='same',
                                         activation=activation_type)
        
        self.conv3 = keras.layers.Conv1D(filters=filters,
                                         kernel_size=kernel_size,
                                         padding='same',
                                         activation=activation_type)
        
        self.dropout1 = keras.layers.SpatialDropout1D(2 * dropout_rate)
        self.dropout2 = keras.layers.SpatialDropout1D(dropout_rate)
        
        self.pooling = keras.layers.MaxPool1D(pool_size)
        
        self.flatten = keras.layers.Flatten()
        
        self.dense1 = keras.layers.Dense(8 * filters)
        self.dense2 = keras.layers.Dense(filters)
        
        self.classify = keras.layers.Dense(2, activation='softmax')
        
    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.dropout1(x, training=training)
        
        x = self.conv2(x)
        x = self.pooling(x)
        x = self.dropout2(x, training=training)
        
        x = self.conv3(x)
        
        x = self.flatten(x)
        
        x = self.dense1(x)
        x = self.dense2(x)
        
        x = self.classify(x)
        
        return x

    def get_config(self):
        config = super(LeNetLike, self).get_config().copy()
        config.update({
            'kernel_size': self.kernel_size,
            'filters': self.filters,
            'pool_size': self.pool_size,
            'activation_type': self.activation_type,
            'dropout_rate': self.dropout_rate,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

@keras.saving.register_keras_serializable()
def customized_loss(true, pred):
    losss = LOSS_FUNCTION(true, pred)
    # save for debugging
    np.save("debug/true_pred.npy", np.hstack([true.numpy(), pred.numpy()]))
    return losss

CUSTOM_OBJECTS_TASKED = {
    'LeNetLike': LeNetLike,
    'customized_loss': customized_loss,
}
