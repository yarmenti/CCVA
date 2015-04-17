# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 15:33:01 2015

@author: Yann
"""

import numpy as np
from itoprocess import ItoProcessPath

class StudentProcess(ItoProcessPath):
    def __init__(self, X_0, df, volatility, time, brownian_correl_matrix=None):
        _ = lambda **x:0
        vol_p = volatility
        if isinstance(vol_p, (list, np.ndarray)):
            def tmp():
                return np.array(volatility)
            vol_p = tmp
                    
        assert df>2, "The df must be greater than 2, in order to have finite variance"
        self._df_ = df
        self._scaling_factor = np.sqrt((self._df_-2)/self._df_)
        
        super(StudentProcess, self).__init__(X_0, _, vol_p, time, brownian_correl_matrix)
        
    def simulate(self):
        return super(StudentProcess, self)._super_simulate_()        
        
    def _compute_differential(self):
        k = np.random.chisquare(self._df_)
        
        zeros = np.zeros(self._correl_matrix.shape[0])
        gaussians = np.random.multivariate_normal(zeros, self._correl_matrix, self.time.size-1)
        
        coeff = np.sqrt(self._df_/k)
        students = coeff*gaussians
        students *= self._scaling_factor
        
        tmp = np.multiply(self._diff_coeff_(), students)
        return np.multiply(tmp.T, self._sqrt_delta_)