# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 16:18:43 2015

@author: Yann
"""

import numpy as np
from ..path import Path
from itoprocess import ItoProcessPath
from brownianmotion import (
    StandardBrownianMotion,
    BrownianMotionWithJumps
)
from studentprocess import StudentProcess

def test_init_pure_deterministic_from_ito_dim2():
    h = 1/360.
    time_grid = Path.generate_time_grid(0, 2, h)
    x0 = [0., 0.]
    
    def drift(**kwargs):
        return np.array([1, -.2])
        
    def vol(**kwargs):
        return 0.
    
    correl = np.array([[1, 1], [1, 1]])    
    
    ito_path = ItoProcessPath(x0, drift, vol, time_grid, correl)
    
    ito_path.plot() 
    
def test_init_pure_brownian_from_ito_dim2():
    h = 1/360.
    time_grid = Path.generate_time_grid(0, 2, h)
    x0 = [0., 0.]
    
    def drift(**kwargs):
        return np.array([0., 0.])
        
    def vol(**kwargs):
        return np.array([1, 1])
    
    correl = np.array([[1, -1], [-1, 1]])    
    
    ito_path = ItoProcessPath(x0, drift, vol, time_grid, correl)
    
    ito_path.plot() 
    
def test_init_brownian_dim3():   
    h = 1/360.
    time_grid = Path.generate_time_grid(0, 2, h)

    vol = [1, 1, 0.1]
        
    w0 = [0., 0., 5.]
    correl = np.array([[1, -1, 0], [-1, 1, 0], [0, 0, 1]])    
    W = StandardBrownianMotion(w0, vol, time_grid, correl)
    W.plot()
    
def test_init_brownian_dim2():    
    h = 1/360.
    time_grid = Path.generate_time_grid(0, 2, h)
    
    def vol(**kwargs):
        return np.array([1, 1])
        
    w0 = [0., 0.]
    correl = np.array([[1, -1], [-1, 1]])    
    W = StandardBrownianMotion(w0, vol, time_grid, correl)
    W.plot()
    
def test_init_student_dim2():    
    h = 1/360.
    time_grid = Path.generate_time_grid(0, 2, h)
    
    def vol(**kwargs):
        return np.array([1, 1])
        
    w0 = [0., 0.]
    correl = np.array([[1, -1], [-1, 1]])   
    nu = 3.2
    W = StudentProcess(w0, nu, vol, time_grid, correl)
    W.plot()
    
def test_init_student_dim3():   
    h = 1/360.
    time_grid = Path.generate_time_grid(0, 2, h)

    vol = [1, 1, 1]
        
    w0 = [0., 0., 0.]
    correl = np.array([[1, -1, 0], [-1, 1, 0], [0, 0, 1]])    
    nu = 2.1
    W = StudentProcess(w0, nu, vol, time_grid, correl)
    W.plot()
    
def test_init_brownian_jump1():
    h=1./360
    time_grid = BrownianMotionWithJumps.generate_time_grid(0, 2, h)

    drift = [0]
    vol = [1]
        
    w0 = [0.]
    
    BJ = BrownianMotionWithJumps(w0, drift, vol, time_grid)
    def_times = [1.1, 1.4]
    jumps = [4, -1]

    BJ.simulate(jumps_times=def_times, jumps_sizes=jumps)
    BJ.plot()    
    
def test_init_brownian_jump2():
    h = 1/360.
    time_grid = BrownianMotionWithJumps.generate_time_grid(0, 2, h)

    drift = [0, 0]
    vol = [1, 1]
        
    w0 = [0., 0.]
    correl = np.array([[1, 1], [1, 1]])  
    
    BJ = BrownianMotionWithJumps(w0, drift, vol, time_grid, correl)
    def_times = [1.1, 1.4]
    jumps = [[4, -4], 
             [-3, 3]]

    BJ.simulate(jumps_times=def_times, jumps_sizes=jumps)
    BJ.plot()
    
def test_init_brownian_rel_jump1():
    h = 1./360
    time_grid = BrownianMotionWithJumps.generate_time_grid(0, 2, h)

    drift = [0]
    vol = [1]
        
    w0 = [100.]
    
    BJ = BrownianMotionWithJumps(w0, drift, vol, time_grid)
    def_times = [1.1, 1.4]
    jumps = [1.01, 0.98]

    BJ.simulate(jumps_times=def_times, jumps_sizes=jumps, relative=True)
    BJ.plot()    
    
def test_init_brownian_rel_jump2():
    h = 1/360.
    time_grid = BrownianMotionWithJumps.generate_time_grid(0, 2, h)

    drift = [0, 0]
    vol = [1, 1]
        
    w0 = [1., 1.]
    correl = np.array([[1, 1], [1, 1]]) 
    
    BJ = BrownianMotionWithJumps(w0, drift, vol, time_grid, correl)
    def_times = [0.34, 1.4]
    jumps = [[3, 1./3], 
             [0.5, 2]]

    BJ.simulate(jumps_times=def_times, jumps_sizes=jumps, relative=True)
    BJ.plot()