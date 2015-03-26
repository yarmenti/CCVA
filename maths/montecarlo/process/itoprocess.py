# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 11:30:38 2015

@author: Yann
"""
import numpy as np
import types

from ..path import Path

class ItoProcessPath(Path):
    def __init__(self, X_0, drift_func, diff_coeff_func, time, brownian_correl_matrix=None, simulate=True):        
        super(ItoProcessPath, self).__init__(X_0, time)
        
        assert isinstance(drift_func, (types.FunctionType, Path)), "The drift_func must be of type function or Path"
        self._is_drift_process = isinstance(drift_func, Path)
        self._drift_coeff_ = drift_func        
        
        assert isinstance(diff_coeff_func, (types.FunctionType, Path)), "The diff_coeff_func must be of type function or Path"
        self._is_diff_coeff_process = isinstance(diff_coeff_func, Path)
        self._diff_coeff_ = diff_coeff_func

        if brownian_correl_matrix is None:
            assert self.dimension == 1, "Processes with dimension >= 1 must provide brownian_correl_matrix"
            self._correl_matrix = np.array([[1]])
        else:
            tmp = np.array(brownian_correl_matrix)
            shape = tmp.shape
            assert(shape[0] == shape[1]), "The brownian_correl_matrix is not squared"
            tr = tmp.transpose()
            assert ((tmp == tr).all()), "The brownian_correl_matrix is not symmetric"
            assert(all(v==1 for v in tmp.diagonal())), "The brownian_correl_matrix diagonal is not ones"
            assert((-1 <= tmp).all() and (tmp <= 1).all()), "The brownian_correl_matrix has absolute correlation higher than one"
            
            self._correl_matrix = tmp
                
        self.init_delta_time()
        self.check_time_drift_and_diff_coeff_consistency()        
        
        if simulate:
            self.simulate()
    
    def set_time(self, time):    
        super(ItoProcessPath, self).set_time(time)
        self.init_delta_time()
    
    def _compute_differential_(self):
        raise NotImplementedError()
        
    def simulate(self):
        if self._is_drift_process:
            self._drift_coeff_.simulate()
        
        if self._is_diff_coeff_process:
            self._diff_coeff_.simulate()        
        
        zeros = np.zeros(self._correl_matrix.shape[0])
        gaussians = np.random.multivariate_normal(zeros, self._correl_matrix, self.time.size-1).T
        
        values = np.ndarray((self.dimension, self.time.size))
        
        values[:, 0] = self._x0_.reshape(self.dimension)
        shifted_time = self.time[1:]
        
        for i, (t, dt, sqrt_dt) in enumerate(zip(shifted_time, self._delta_, self._sqrt_delta_)):
            d = self._drift_coeff_(t=t, index_time=i, x=values[:, i]) * dt
            vol = self._diff_coeff_(t=t, index_time=i, x=values[:, i]) * sqrt_dt * gaussians[:, i]
            
            dXt = d+vol             
            values[:, i+1] = values[:, i] + dXt
                        
        self._values_ = np.array(values)
            
    def check_time_drift_and_diff_coeff_consistency(self):
        if self._is_drift_process:
            drift_diff = np.setdiff1d(self._drift_.time, self.time)
            assert drift_diff.size == 0, "The time_grid of the drift must be contained in self.time"
            
        if self._is_diff_coeff_process:
            diff_coeff_diff = np.setdiff1d(self._diff_coeff_.time, self.time)
            assert diff_coeff_diff.size == 0, "The time_grid of the diff_coeff must be contained in self.time"
            
    def init_delta_time(self):
        self._delta_ = np.ediff1d(self.time)
        self._sqrt_delta_ = np.sqrt(self._delta_)
        
    def _super_simulate_(self):
        return super(ItoProcessPath, self).simulate()
        
Path.register(ItoProcessPath)