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
                 config,
                 ):
        self._config = config

    @property
    def sample_num(self):
        return self._config._sample_num

    def ReadData(self):
        #Todo
        image = np.zeros([1])
        image.reshape(self._config.shape)
        return image

    def ReadLabel(self):
        #Todo
        label = np.zeros([1])
        return label

    def DataTransform(self, image):
        #Todo
        return image

    def LabelTransform(self, label):
        # Todo
        return label

    def NextBatch(self):
        #Todo
        image = self.ReadData()
        image = self.DataTransform(image)
        label = self.ReadLabel()
        return image, label