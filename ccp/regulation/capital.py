# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 15:39:46 2015

@author: Yann
"""

import numpy as np

class RegulatoryCapital(object):
    def __init__(self, beta, im_accounts, df_accounts, sig, portfolio, **kwargs):        
        cm_nb = df_accounts._size_
        
        self._df_ = df_accounts
        self._im_ = im_accounts
        
        self._eqty_ = sig
        self._coeff_ = 1 + beta*(cm_nb/(cm_nb - 2))
        self._port_ = portfolio
        
        self._init_kwargs_(**kwargs)
        
    def _init_kwargs_(self, **kwargs):
        self._ead_coeff_ = kwargs.get('ead_coeff', 1.)        
        
    def _compute_k_cms_(self, t):
        E = self._eqty_()
        df_tot = self._df_.total_default_fund()        
        df_prime_cm = np.maximum(df_tot - 2*self._df_.mean_contribution(), 0)
        df_prime = E + df_prime_cm
        
        k_ccp = self.compute_k_ccp(t)
        
        c1 = 0.0016
        if df_prime > 0:
            c1 = np.maximum(0.0016, 0.016*(k_ccp/df_prime)**0.3)
        
        c2 = 1.0
        mu = 1.2
        
        res = 0
        if df_prime < k_ccp:
            res = c2 * (mu*(k_ccp-df_prime) + df_prime_cm)
        elif E < k_ccp and k_ccp < df_prime:
            res = c2*(k_ccp-E) + c1*(df_prime-k_ccp)
        elif k_ccp <= E:
            res = c1*df_prime_cm
            
        return res        
        
    def compute_k_cm(self, index, t):
        total_df = self._df_.total_default_fund()
        if total_df <= 0:
            raise RuntimeError("The total default fund must be > 0")
        
        k_cms = self._compute_k_cms_(t)
        
        df = self._df_.get_amount(index)
        
        return self._coeff_ * df/total_df * k_cms
    
    def compute_k_ccp(self, t):
        capital_ratio = 0.08
        risk_weight = 0.2
        
        ebrms = self._port_.compute_exposure(t)
                
        ims = self._im_.amounts
        dfs = self._df_.amounts
        states = self._im_.states.alive_states
        
        res = 0
        for ebrm, im, df, is_alive in zip (ebrms.flat, ims, dfs, states):            
            if is_alive:
                mod_ebrm = ebrm * self._ead_coeff_                
                #print mod_ebrm, "_", im+df
                res += np.maximum(mod_ebrm - im - df, 0)
        
        res *= capital_ratio * risk_weight
        
        return res
    
    