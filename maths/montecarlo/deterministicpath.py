# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 11:30:38 2015

@author: Yann
"""

import numpy as np
from path import Path

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
        
        self._values_ = np.array(vals).transpose()    
                
    def simulate(self):
        pass
    
    def _compute_differential_(self):
        raise NotImplementedError()
        
Path.register(DeterministicPath)