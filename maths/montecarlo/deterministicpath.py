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
        vals = []
        for t in time:
            vals.append(f(t))
        
        self._values_ = np.array(vals).transpose()
                
    def simulate(self):
        pass
    
    def _compute_differential_(self):
        raise NotImplementedError()
        
Path.register(DeterministicPath)