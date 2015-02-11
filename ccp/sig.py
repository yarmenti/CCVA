# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 10:41:53 2015

@author: Yann
"""

import numpy as np

class SkinInTheGame(object):
    def __init__(self, initial_value):
        assert (initial_value >= 0), "The initial value is not positive"
        self._init_val_ = initial_value
        self._val_ = initial_value
        
    def handle_breach(self, breach):
        jump = -np.minimum(breach, self._val_)
        self._val_ += jump        
        self._val_ = np.maximum(self._val_, 0)
        return breach+jump
    
    def __call__(self):
        return self._val_
    
    def recover(self, val=None):
        self._val_ = self._init_val_ if val is None else val