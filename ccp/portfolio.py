# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 11:17:23 2015

@author: Yann
"""

import numpy as np

from finance.portfolio import Portfolio

class CCPPortfolio(Portfolio):
    def __init__(self, members_positions_mat, derivatives, prices, exposures):
        # The shape = (m,n) with m the number of CMs, n the number of assets
        mat = np.matrix(members_positions_mat)
        for i in range(mat.shape[1]):
            assert np.sum(mat[:, i]) == 0, "The total portfolio composition is not neutral"
            
        ccp_positions = -mat
        super(CCPPortfolio, self).__init__(ccp_positions, derivatives, prices)
            
        self._exposures_ = np.array(exposures)
                
        self._nb_cm_, _ = self.weights.shape
        
    def compute_exposure(self, t, **kwargs):
        tmp = np.zeros((self.cm_number, 1))
        for i, e in enumerate(self._exposures_):
            tmp += e(t=t, portfolio=self, derivative_index=i, **kwargs)
                
        return tmp
            
    def get_exposure(self, i):
        return self._exposures_[i]
            
    def set_exposure(self, i, value):
        self._exposures_[i] = value
            
    @property
    def cm_number(self):
        return self._nb_cm_