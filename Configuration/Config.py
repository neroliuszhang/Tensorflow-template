from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

class DataConfig(object):
    def __init__(self,
                img_shape,  # w*h*ch
                train_path,
                train_num,
                train_label_path,
                valid_path,
                valid_num,
                valid_label_path,
                eval_path,
                eval_num,
                eval_label_path,

                model_save_path,
                ):
        self._shape = img_shape
        self._train_path = train_path
        self._valid_path = valid_path
        self._eval_path = eval_path

class TrainConfig(object):
    def __init__(self,
                 optimizor = 'sgd',
                 learn_rate = 0.001,
                 batch_size = 10,
                 epoch_num = 1000
                 ):
        self._optimizor = optimizor
        self._learn_rate = learn_rate
        self._batch_size = batch_size
        self._epoch_num = epoch_num

class Config(object):
    def __init__(self,
                 data_config,
                 train_config
                 ):
        self._data_config = data_config
        self._train_config = train_config