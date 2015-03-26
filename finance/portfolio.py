# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 10:55:50 2015

@author: Yann
"""

import numpy as np

class Portfolio(object):    
    def __init__(self, matrix_positions, derivatives, prices, exposures):
        mat = matrix_positions if isinstance(matrix_positions, np.ndarray) else np.matrix(matrix_positions)        
        
        self._derivatives_ = np.array(derivatives)
        assert mat.shape[1] == self._derivatives_.size, "The dimensions of the positions and the underlyings do not match"
        
        self._prices_ = np.array(prices)
        assert mat.shape[1] == self._prices_.size, "The dimensions of the positions and the prices do not match"
                
        self._port_amounts_ = (np.absolute(mat)).sum(axis = 1).reshape(1, mat.shape[0])
        self._positions_ = np.divide(mat.astype(float), self._port_amounts_.T)
        
        self._exposures_ = np.array(exposures)
        self._nb_contractors_, _ = self.weights.shape
        
    def compute_pl(self, assets_p_and_l):
        tmp = np.dot(self._positions_, assets_p_and_l)  
        res = np.multiply(self.amounts, tmp)        
        return res
            
    @property
    def derivatives(self):
        return self._derivatives_
    
    @property
    def amounts(self):
        return self._port_amounts_[0].tolist()
        
    @property
    def weights(self):
        return self._positions_
        
    def compute_exposure(self, t, **kwargs):
        tmp = np.zeros((self._nb_contractors_, 1))
        for e in self._exposures_:
            temp = e(t=t, portfolio=self, **kwargs)            
            tmp += temp
                
        return np.multiply(tmp, self._port_amounts_.T)
            
    def get_exposure(self, i):
        return self._exposures_[i]
            
    def set_exposure(self, i, value):
        self._exposures_[i] = value