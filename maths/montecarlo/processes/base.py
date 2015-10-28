import numpy as np
import matplotlib.pyplot as plt

from abc import ABCMeta, abstractmethod


class Process(object):
    __metaclass__ = ABCMeta

    def __init__(self, time_arr, x_0):
        self.__values = None
        self.__time = None

        if isinstance(x_0, list):
            x_0 = np.array(x_0, copy=True)

        if not isinstance(x_0, np.ndarray):
            raise ValueError("The X_0 value must be of type np.array")

        if x_0.ndim == 1:
            x_0 = x_0.reshape((x_0.shape[0], 1))

        if x_0.ndim != 2:
            raise ValueError("The number of dimensions of X_0 must be equal to 2, given: %d"%x_0.ndim)

        if x_0.shape[1] != 1:
            raise ValueError("X_0 must have only one column, given: %s"%x_0.shape[1])

        self._x0 = x_0
        self.__dim = self._x0.size

        self.time = time_arr
        self.__values = np.zeros((self.dimension, self.time.shape[0]))

    def __call__(self, t):
        if t < 0:
            raise ValueError("t must be positive")

        if t not in self.time:
            raise ValueError("t must be in the time_grid, given: %s.\n Time grid = %s"%(t, self.time))

        index = np.where(self.time == t)[0]
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
    def conditional_expectation(self, T, t):
        pass

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