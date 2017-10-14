'''
Auther:         Nero.Zhang
Version:        0.0
Description:
                1 Dataset template for a simple and flexible wrapper for tensorflow.
                2 Dataset reads sample data and label from data source,
                  transforms data and label(for example data enhancement and label one hot transform),
                  and returns data and label to caller.
                3 Dataset is the input of model, user should implement the class according to training
                  dataset.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import collections
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed

class Dataset(object):
    def __init__(self,
                 name,
                 config,
                 ):
        self._name = name
        self._config = config

    @property
    def sample_num(self):
        return self._config._sample_num

    def __ReadData(self):
        #Todo
        image = np.zeros([1])
        image.reshape(self._config.shape)
        return image

    def __ReadLabel(self):
        #Todo
        label = np.zeros([1])
        return label

    def __DataTransform(self, image):
        #Todo
        return image

    def __LabelTransform(self, label):
        # Todo
        return label

    def NextBatch(self):
        #Todo
        image = self.__ReadData()
        image = self.__DataTransform(image)
        label = self.__ReadLabel()
        return image, label