# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 10:55:50 2015

@author: Yann
"""

import numpy as np

class Portfolio(object):    
    def __init__(self, matrix_positions, derivatives, prices, exposures):
        mat = np.matrix(matrix_positions)        
        
        self._derivatives_ = np.array(derivatives)
        assert mat.shape[1] == self._derivatives_.size, "The dimensions of the positions and the underlyings do not match"
        
        self._prices_ = np.array(prices)
        assert mat.shape[1] == self._prices_.size, "The dimensions of the positions and the prices do not match"
                
        self._port_amounts_ = (np.absolute(mat)).sum(axis = 1).T
        self._positions_ = mat.astype(float) / np.sum(np.absolute(mat), axis=1)
        
        self._exposures_ = np.array(exposures)
        self._nb_contractors_, _ = self.weights.shape
        
    def compute_pl(self, assets_p_and_l):
        tmp = np.dot(self._positions_, assets_p_and_l)        
        res = np.multiply(self._port_amounts_, tmp)        
        return np.array(res.A1)
            
    @property
    def derivatives(self):
        return self._derivatives_
    
    @property
    def amounts(self):
        return self._port_amounts_
    
    @property
    def weights(self):
        return self._positions_
        
    def compute_exposure(self, t, **kwargs):
        tmp = np.zeros((self._nb_contractors_, 1))
        for i, e in enumerate(self._exposures_):
            tmp += e(t=t, portfolio=self, **kwargs)
                
        return tmp
            
    def get_exposure(self, i):
        return self._exposures_[i]
            
    def set_exposure(self, i, value):
        self._exposures_[i] = value