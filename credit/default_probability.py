# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 12:42:15 2015

@author: Yann
"""

import abc
import numpy as np

class DefaultableModel(object):
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractmethod
    def survival_proba(self, t):
        pass

    def default_proba(self, t):
        return 1. - self.survival_proba(t)

class FlatIntensity(DefaultableModel):
    def __init__(self, intensity):
        self.__lambda__ = intensity
        
    def survival_proba(self, t):
        return np.exp(-self.__lambda__*t)
        
DefaultableModel.register(FlatIntensity)