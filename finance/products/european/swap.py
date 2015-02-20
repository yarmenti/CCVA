# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 15:37:27 2015

@author: Yann
"""

import numpy as np
from scipy import interpolate

from .european import EuropeanContract

class SwapContract(EuropeanContract):
    def __init__(self, underlying, df_process, dates, underlying_index=0):
        super(SwapContract, self).__init__(underlying, dates[-1], df_process, underlying_index)
        
        self._pillars_ = np.sort(dates)
        self._strike_ = self.compute_strike(dates[0])

    @property
    def delta_time(self):
        return self._delta_

    @property
    def strike(self):
        return self._strike_
        
    @property
    def pillars(self):
        return self._pillars_
    
    def compute_strike(self, t):
        self._v_df_ = np.vectorize(self._df_)        
        self._pillar_df_ = self._v_df_(self._pillars_)
        self._delta_ = np.ediff1d(self._pillars_)
        
        den = np.dot(self._delta_, self._pillar_df_[1:])
        
        ratio_df = self._pillar_df_[1:] / self._pillar_df_[:-1]
        num = np.dot(self._delta_, ratio_df)
        
        return self._get_St_(t)*num/den
        
    def price(self, t):        
        fst_payment_idx = np.searchsorted(self._pillars_, t, side='right')
        if fst_payment_idx >= len(self._pillars_):
            return 0.
        
        df_t = self.discount_factor(t)
        
        t_i_star_m_1 = self._pillars_[fst_payment_idx-1]
        S_t_i_star_m_1 = self._get_St_(t_i_star_m_1)
        
        first_coupon = self._delta_[fst_payment_idx-1]
        first_coupon *= self._pillar_df_[fst_payment_idx]
        first_coupon *= S_t_i_star_m_1 - self.strike
        first_coupon /= df_t
        
        deltas = self._delta_[fst_payment_idx:]
        ratio_df = self._pillar_df_[fst_payment_idx + 1:] / self._pillar_df_[fst_payment_idx:-1]
        St = self._get_St_(t)

        term1 = np.dot(deltas, ratio_df)*St
        
        term2 = np.dot(deltas, self._pillar_df_[fst_payment_idx + 1:])
        term2 *= -self._strike_/df_t    

        return first_coupon + term1 + term2
      
    def __str__(self):
        pill = ("{" +', '.join(['%.2f']*len(self.pillars))+"}")%tuple(self.pillars)
        return "Swap contract of maturity T = %d years, over S^%d with strike K = %.3f, paying at %s"%(self.maturity, self._underlying_index_, self.strike, pill)
    
    def __additional_points_subprocess__(self, **kwargs):
        t = kwargs['t']
        t_ph = kwargs['t_ph']
        
        pill_i = (self._pillars_ <= t_ph)
        pills = self._pillars_[pill_i]
        
        tmp = {t_: self._get_St_(t_) for t_ in pills}

        special_pill_i = (t < self._pillars_) & (self._pillars_ <= t_ph)
        if special_pill_i.any():
            special_pills = self._pillars_[special_pill_i]
                        
            time = [t, t_ph]
            
            current = kwargs['current']
            S = [current[t], current[t_ph]]

            f = interpolate.interp1d(time, S)  
            tmp.update({t_: f(t_) for t_ in special_pills})
                
        return tmp
    
    @classmethod
    def generate_payment_dates(cls, first_date, maturity, step):
        res = np.arange(first_date, maturity+step, step)
        if res[-1] > maturity:
            res = np.delete(res, -1)
            
        return res
    
EuropeanContract.register(SwapContract)

#####################################################################