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
        
        self.__pills = np.sort(dates)

        self.__vect_df = np.vectorize(self.__df)
        self.__discounted_pills = self.__vect_df(self.__pills)
        self.__delta_pills = np.ediff1d(self.__pills)

        self.__k = self.compute_strike(dates[0])

    @property
    def delta_time(self):
        return np.array(self.__delta_pills, copy=True)

    @property
    def strike(self):
        return self.__k
        
    @property
    def pillars(self):
        return self.__pills
    
    def compute_strike(self, t):
        den = np.dot(self.__delta_pills, self.__discounted_pills[1:])
        
        ratio_df = self.__discounted_pills[1:] / self.__discounted_pills[:-1]
        num = np.dot(self.__delta_pills, ratio_df)
        
        return self.S(t)*num/den
        
    def price(self, t):
        fst_payment_idx = np.searchsorted(self.__pills, t, side='right')
        if fst_payment_idx >= len(self.__pills):
            return 0.
        
        df_t = self.discount_factor(t)
        
        t_i_star_m_1 = self.__pills[fst_payment_idx-1]
        st_i_star_m_1 = self.S(t_i_star_m_1)
        
        first_coupon = self.__delta_pills[fst_payment_idx-1]
        first_coupon *= self.__discounted_pills[fst_payment_idx]
        first_coupon *= st_i_star_m_1 - self.strike
        first_coupon /= df_t
        
        deltas = self.__delta_pills[fst_payment_idx:]
        ratio_df = self.__discounted_pills[fst_payment_idx + 1:] / self.__discounted_pills[fst_payment_idx:-1]
        st = self.S(t)

        term1 = np.dot(deltas, ratio_df)*st
        
        term2 = np.dot(deltas, self.__discounted_pills[fst_payment_idx + 1:])
        term2 *= -self.__k/df_t

        return first_coupon + term1 + term2
      
    def __str__(self):
        pill = ("{" + ', '.join(['%.2f']*len(self.pillars))+"}")%tuple(self.pillars)
        return "Swap contract of maturity T = %d years, over S^%d with strike K = %.3f, paying at %s"%(self.maturity, self.__udlyg_idx, self.strike, pill)
    
    def __additional_points_subprocess__(self, **kwargs):
        t = kwargs['t']
        t_ph = kwargs['t_ph']
        
        last_pill_idx = np.searchsorted(self.pillars, t) - 1
        last_pill = self.pillars[last_pill_idx]
        tmp = {last_pill: self.S(last_pill)}

        special_pill_i = (t < self.__pills) & (self.__pills <= t_ph)
        if special_pill_i.any():
            special_pills = self.__pills[special_pill_i]
                        
            time = [t, t_ph]
            
            current = kwargs['current']
            s = [current[t], current[t_ph]]

            f = interpolate.interp1d(time, s)
            tmp.update({t_: f(t_) for t_ in special_pills})
                
        return tmp
    
    @classmethod
    def generate_payment_dates(cls, first_date, maturity, step):
        res = np.arange(first_date, maturity+step, step)
        if res[-1] > maturity:
            res = np.delete(res, -1)
            
        return res
    
EuropeanContract.register(SwapContract)