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

        if not isinstance(X_0, np.ndarray):
            raise ValueError("The X_0 value must be of type np.array")

        if X_0.ndim != 2:
            raise ValueError("The number of dimensions of X_0 must be equal to 2, given: %d"%X_0.ndim)

        nbr, nbc = X_0.shape
        self._x0 = X_0.T if nbr == 1 else X_0

        if not isinstance(time_arr, (list, np.ndarray)):
            raise ValueError("The time must be of type list or np.array")

        self.__time = np.array(time_arr)
        self.__dim = self._x0.size
        self.__vals = None

        if simulate:
            self.simulate()
    
    @property
    def time(self):
        return np.array(self.__time, copy=True)

    @time.setter
    def time(self, time):
        assert isinstance(time, (list, np.ndarray)), "The time must be of type list or np.array"
        self.__time = np.array(time)

    @property    
    def values(self):
        if self.__vals is None:
            return None

        return np.array(self.__vals, copy=True)

    @values.setter
    def values(self, vals):
        tmp_vals = np.array(vals, copy=True)
        actual_vals = self.values

        if actual_vals is not None:
            if tmp_vals.shape != actual_vals.shape:
                raise ValueError("The shape between old values and new ones do not match. "
                                 "Given %s, old %s"%(tmp_vals.shape,actual_vals.shape ))

            if tmp_vals.size != actual_vals.size:
                raise ValueError("The size between old values and new ones do not match")

        self.__vals = tmp_vals

    @property
    def dimension(self):
        return self.__dim
    
    @abstractmethod
    def _compute_differential(self):
        pass

    def simulate(self):
        differentials = self._compute_differential()
        differentials = np.hstack((self._x0, differentials))
        self.__vals = np.cumsum(differentials, axis=1)
    
    def __call__(self, t):
        if t < 0:
            raise ValueError("t must be positive")

        if t not in self.time:
            raise ValueError("t must be in the time_grid, given: %s.\n Time grid = %s"%(t, self.time))

        index = np.where(self.time==t)
        
        return self.__vals[:, index]
            
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

######################################
# Version V2 de Path => Process


class Process(object):
    __metaclass__ = ABCMeta

    __values = None
    __time = None

    def __init__(self, time_arr, X_0):
        if isinstance(X_0, list):
            X_0 = np.array(X_0, copy=True)

        if not isinstance(X_0, np.ndarray):
            raise ValueError("The X_0 value must be of type np.array")

        if X_0.ndim == 1:
            X_0 = X_0.reshape((X_0.shape[0], 1))

        if X_0.ndim != 2:
            raise ValueError("The number of dimensions of X_0 must be equal to 2, given: %d"%X_0.ndim)

        if X_0.shape[1] != 1:
            raise ValueError("X_0 must have only one column, given: %s"%X_0.shape[1])

        self._x0 = X_0
        self.__dim = self._x0.size

        self.time = time_arr
        self.__values = np.zeros((self.dimension, self.time.shape[0]))

    def __call__(self, t):
        if t < 0:
            raise ValueError("t must be positive")

        if t not in self.time:
            raise ValueError("t must be in the time_grid, given: %s.\n Time grid = %s"%(t, self.time))

        index = np.where(self.time==t)[0]
        return self.values[:, index]

    @abstractmethod
    def _time_set(self):
        pass

    @property
    def time(self):
        return np.array(self.__time, copy=True)

    @time.setter
    def time(self, time):
        if not isinstance(time, (list, np.ndarray)):
            raise ValueError("The time must be of type list or np.array")

        time = list(set(time))
        self.__time = np.array(time)
        self.__time.sort()

        self._time_set()

    @property
    def values(self):
        return np.array(self.__values, copy=True)

    @values.setter
    def values(self, vals):
        if not isinstance(vals, (list, np.ndarray)):
            raise ValueError("The vals must be of type list or np.array")

        tmp_vals = np.array(vals)
        if tmp_vals.ndim == 1:
            tmp_vals = tmp_vals.reshape((1, tmp_vals.shape[0]))

        if tmp_vals.shape[0] != self.dimension:
            raise ValueError("The number of dimension (%s) do not match (given %s)"%(self.dimension, tmp_vals.shape[0]))

        if self.__time.shape[0] != tmp_vals.shape[1]:
            raise ValueError("The shape between time (%s) and vals (%s) do not match."%(self.__time.shape[0], tmp_vals.shape[1]))

        self.__values = tmp_vals

    @property
    def dimension(self):
        return self.__dim

    @abstractmethod
    def simulate(self):
        pass

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