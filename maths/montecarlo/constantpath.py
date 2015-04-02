# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 11:30:38 2015

@author: Yann
"""
import numpy as np
from path import Path

class ConstantPath(Path):
    def __init__(self, X_0, time):        
        super(ConstantPath, self).__init__(X_0, time, True)
        
        self._values_ = self.__compute_values__()        
        
    def _compute_differential_(self):
        return np.zeros((self.dimension, self.time.size - 1))

    def simulate(self):
        pass

    def __compute_values__(self):
        res = np.zeros((self.dimension, self.time.size))
        for i in range(self.dimension):
            res[i, :] = self._x0_[i]
        
        return res
        
Path.register(ConstantPath)