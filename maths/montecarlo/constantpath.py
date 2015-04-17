# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 11:30:38 2015

@author: Yann
"""
import numpy as np
from path import Path, Process


class ConstantPathOld(Path):
    def __init__(self, X_0, time):        
        super(ConstantPathOld, self).__init__(X_0, time, False)
        self.values = self.__compute_values()
        
    def _compute_differential(self):
        return np.zeros((self.dimension, self.time.size - 1))

    def simulate(self):
        pass

    @property
    def time(self):
        return super(ConstantPathOld, self).time

    @time.setter
    def time(self, time):
        super(ConstantPathOld, self.__class__).time.fset(self, time)
        self.values = self.__compute_values()

    def __compute_values(self):
        res = np.zeros((self.dimension, self.time.size))
        for i in range(self.dimension):
            res[i, :] = self._x0[i]
        
        return res

Path.register(ConstantPathOld)

#########################################################################

class ConstantProcess(Process):
    def __init__(self, time, X_0):
        super(ConstantProcess, self).__init__(time, X_0)
        self.simulate()

    def _time_set(self):
        self.simulate()

    def simulate(self):
        vals = np.tile(self._x0, self.time.shape)
        super(ConstantProcess, self.__class__).values.fset(self, vals)

Process.register(ConstantProcess)

