# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 11:30:38 2015

@author: Yann
"""

import numpy as np
from path import Path, Process


class DeterministicPath(Path):
    def __init__(self, f, time):        
        super(DeterministicPath, self).__init__(f(0), time, False)
        self.__func = f
        
        self.__compute_vals(self.time)
                
    def set_time(self, time):
        super(DeterministicPath, self).set_time(time)
        self.__compute_vals(self.time)
                
    def __compute_vals(self, time):
        vals = []
        for t in time:
            vals.append(self.__func(t))
        
        self.__vals = np.array(vals).transpose()
                
    def simulate(self):
        pass
    
    def _compute_differential(self):
        raise NotImplementedError()

Path.register(DeterministicPath)

#######################################


class HistoricalProcess(Process):
    def __init__(self, time, values):
        if isinstance(values, list):
            values = np.array(values)

        x0 = None
        if values.ndim == 1:
            x0 = [values[0]]
        elif values.ndim == 2:
            x0 = values[:, 0]
        else:
            raise ValueError("The values must be of dimension 2 or 1, given: %s"%values.ndim)

        super(HistoricalProcess, self).__init__(time, x0)
        self.values = values

        def func():
            raise NotImplementedError("Cannot change time for historical process")

        self._time_set = func

    def _time_set(self):
        pass

    def simulate(self):
        raise NotImplementedError("Cannot simulate historical process")

Process.register(HistoricalProcess)