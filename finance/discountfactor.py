# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 15:25:10 2015

@author: Yann
"""

import numpy as np
from abc import ABCMeta, abstractmethod

class DiscountFactor(object):
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def __call__(time):
        pass
    
class ConstantRateDiscountFactor(DiscountFactor):
    def __init__(self, rate):
        self._r_ = rate
        
    def __call__(self, t):
        return np.exp(-self._r_*t)
        
    def __str__(self):
        return "Constant discount factor process with rate r = %.2f"%(self._r_)
        
DiscountFactor.register(ConstantRateDiscountFactor)