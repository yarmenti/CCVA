# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 15:37:27 2015

@author: Yann
"""

from .european import EuropeanContract

class ForwardContract(EuropeanContract):
    def __init__(self, underlying, df_process, maturity, strike, underlying_index=0):
        super(ForwardContract, self).__init__(underlying, maturity, df_process, underlying_index)        
        self._df_mat_ = self._df_(self._T_)
        
        self._K_ = strike
        
    def price(self, t):
        if t > self._T_:
            return 0

        S_t = self._get_St_(t)
        res = S_t - self._df_mat_/self._df_(t) * self._K_
        return res
        
    @property
    def strike(self):
        return self._K_
        
    def __str__(self):
        return "Forward contract of strike K = %d and maturity T = %d years over S^%d"%(self.strike, self.maturity, self._underlying_index_)
    
EuropeanContract.register(ForwardContract)

#####################################################################

from scipy.stats import norm
import numpy as np

def ind_forward_brownian_quantile(t, h, drift_t, vol_t, alpha, forward, weight, df):
    if t > forward.maturity:
        return 0.
        
    if t+h > forward.maturity:
        hp = forward.maturity - t
        return ind_forward_brownian_quantile(t, hp, drift_t, vol_t, alpha, forward, weight, df)
    
    K = forward.strike
    
    df_mat = df(forward.maturity)
    df_t = df(t)
    df_tph = df(t+h)
    
    conf_level = alpha if weight>0 else 1-alpha
    quantile_inv = norm.ppf(conf_level)
    
    delta_S = (drift_t*h) + vol_t*np.sqrt(h)*quantile_inv 
    
    return weight*(delta_S + K*df_mat*(1./df_t - 1./df_tph))