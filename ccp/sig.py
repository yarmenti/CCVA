# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 10:41:53 2015

@author: Yann
"""

import numpy as np

class SkinInTheGame(object):
    def __init__(self, initial_value):
        assert (initial_value >= 0), "The initial value is not positive"
        self.__init_val = initial_value
        self.__val = initial_value
        
    def handle_breach(self, breach):
        jump = -np.minimum(breach, self.__val)
        self.__val += jump
        self.__val = np.maximum(self.__val, 0)
        return breach+jump

    @property
    def value(self):
        return self.__val
    
    def recover(self, val=None):
        self.__val = self.__init_val if val is None else val