# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 15:33:01 2015

@author: Yann
"""

import numpy as np
from itoprocess import ItoProcessPath

class BrownianMotion(ItoProcessPath):
    def __init__(self, X_0, drift, volatility, time, brownian_correl_matrix=None, simulate=True):        
        vol_p = volatility
        if isinstance(vol_p, (list, np.ndarray)):
            def tmp():
                return np.array(volatility)
            vol_p = tmp
        
        drift_p = drift
        if isinstance(drift_p, (list, np.ndarray)):
            def tmp2():
                return np.array(drift)
            drift_p = tmp2
        
        super(BrownianMotion, self).__init__(X_0, drift_p, vol_p, time, brownian_correl_matrix, False)
        self._m_delta_ = np.tile(self._delta_, (self.dimension, 1)).T
        
        if simulate:
            self.simulate()
        
    def simulate(self):
        return super(BrownianMotion, self)._super_simulate_()    

    def _compute_differential_(self):
        zeros = np.zeros(self._correl_matrix.shape[0])
        gaussians = np.random.multivariate_normal(zeros, self._correl_matrix, self.time.size-1)                
        tmp = np.multiply(self._diff_coeff_(), gaussians)
        tmp = np.multiply(tmp.T, self._sqrt_delta_)
                
        tmp2 = np.multiply(self._drift_coeff_(), self._m_delta_)
        return tmp + tmp2.T

###############################################################################

class StandardBrownianMotion(BrownianMotion):
    def __init__(self, X_0, volatility, time, brownian_correl_matrix=None):
        _ = lambda **x:0        
        super(StandardBrownianMotion, self).__init__(X_0, _, volatility, time, brownian_correl_matrix)        
        
    def _compute_differential_(self):
        zeros = np.zeros(self._correl_matrix.shape[0])
        gaussians = np.random.multivariate_normal(zeros, self._correl_matrix, self.time.size-1)
                
        tmp = np.multiply(self._diff_coeff_(), gaussians)
        return np.multiply(tmp.T, self._sqrt_delta_)
        
###############################################################################
        
class BrownianMotionWithJumps(BrownianMotion):
    def __init__(self, X_0, drift, volatility, time, brownian_correl_matrix=None):        
        super(BrownianMotionWithJumps, self).__init__(X_0, drift, volatility, time, brownian_correl_matrix, False)  
        
        self._add_rel_jump_ = lambda x, y: np.multiply(x.T, y).T
        self._add_jump_ = lambda x, y: x + np.tile(y.T, (1, x.shape[1]))
        
    def simulate(self, **kwargs):
        super(BrownianMotion, self)._super_simulate_()
        jumps_times = kwargs.get('jumps_times', None)
        jumps_sizes = kwargs.get('jumps_sizes', None)
        relative_jump = kwargs.get('relative', False)
        
        if not (jumps_times and jumps_sizes) :
            return
            
        jumps_sizes = np.matrix(jumps_sizes).T
        
        add_jump = self._add_rel_jump_ if relative_jump else self._add_jump_
        
        for (t, j) in zip(jumps_times, jumps_sizes):
            index = np.argmax(self.time>=t)
            self._values_[:, index:] = add_jump(self._values_[:, index:], j)            