# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 15:59:53 2015

@author: Yann
"""

from path import Path
from constantpath import ConstantPath
from deterministicpath import DeterministicPath

import numpy as np

def test_time_grid_generation():
    Path.generate_time_grid(0, 2, 0.4)
        
def test_init_constant_path():
    time_grid = Path.generate_time_grid(0, 2, 0.2)
    x0 = [0.3, 0.35, 0.4]
    
    constant_path = ConstantPath(x0, time_grid)
        
    constant_path.plot()
    
def test_init_deterministic_path():
    time_grid = Path.generate_time_grid(0, 2, 0.5)

    def f(t):
        return [(t+2)**2, t**2]

    det_path = DeterministicPath(f, time_grid)
    det_path.plot()