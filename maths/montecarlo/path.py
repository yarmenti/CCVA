# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 11:07:58 2015

@author: Yann
"""
import numpy as np
import matplotlib.pyplot as plt

from abc import ABCMeta, abstractmethod

class Path(object):
    __metaclass__ = ABCMeta
    
    def __init__(self, X_0, time_arr, simulate=False):
        """
        X_0 must be a vector
        """
        if isinstance(X_0, (int, long, float)):
            X_0 = np.array([[X_0]])
        
        if isinstance(X_0, list):
            X_0 = np.array(X_0)[np.newaxis]
                
        assert isinstance(X_0, np.ndarray), "The X_0 value must be of type list or np.array"        

        assert X_0.ndim == 2, "The number of dimensions of X_0 must be equal to 2, given: %d"%X_0.ndim
        nbr, nbc = X_0.shape
        self._x0_ = X_0.T if nbr == 1 else X_0
        
        assert isinstance(time_arr, (list, np.ndarray)), "The time must be of type list or np.array"
        self._time_ = np.array(time_arr)

        self._dim_ = self._x0_.size

        if simulate:
            self.simulate()
    
    @property
    def time(self):
        return self._time_
    
    @property    
    def values(self):
        return self._values_
        
    @property
    def dimension(self):
        return self._dim_
    
    @abstractmethod
    def _compute_differential_(self):
        pass
        
    def set_time(self, time):
        assert isinstance(time, (list, np.ndarray)), "The time must be of type list or np.array"
        self._time_ = np.array(time)
    
    def simulate(self):
        differentials = self._compute_differential_()
        differentials = np.hstack((self._x0_, differentials))
        self._values_ = np.cumsum(differentials, axis=1)
    
    def __call__(self, t):
        assert (t >= 0), "t must be positive" 
        assert (t in self.time), "The value of t must be in the time_grid, given: %s. Time grid = %s"%(t, self.time)
        index = np.where(self.time==t)
        
        return self._values_[:, index]        
            
    def plot(self):        
        for i in range(self.dimension):
            plt.plot(self.time, self.values[i, :], )
            
        plt.show()

    @classmethod
    def generate_time_grid(cls, init_val=0, last_val=1, step=0.1, incl_vals=None):
        grid = np.arange(init_val, last_val+step, step)
        if incl_vals is not None:
            tmp = np.append(grid, incl_vals)
            tmp = np.unique(tmp)
            grid = np.sort(tmp)
        
        return grid      