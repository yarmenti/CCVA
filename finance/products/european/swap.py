import numpy as np
from pandas.tslib import _Timedelta
from scipy import interpolate

from .european import EuropeanContract, ContractType


class SwapContract(EuropeanContract):
    def __init__(self, underlying, df_process, dates, underlying_index=0, use_1st_coupon_for_neutr_strike=False):
        super(SwapContract, self).__init__(underlying, dates[-1], df_process, underlying_index)
        
        self.__pills = np.sort(dates)

        self.__vect_df = np.vectorize(self.discount_factor)
        self.__discounted_pills = self.__vect_df(self.__pills)
        self.__delta_pills = np.ediff1d(self.__pills)

        self.__k = self.compute_neutral_strike(use_1st_coupon_for_neutr_strike)

    @property
    def delta_time(self):
        return np.array(self.__delta_pills, copy=True)

    @property
    def strike(self):
        return self.__k
        
    @property
    def pillars(self):
        return np.array(self.__pills, copy=True)

    @property
    def asset_class(self):
        return ContractType.interest_rate

    def compute_neutral_strike(self, use_1st_coupon_for_neutr_strike):
        index, delta_pills_ind = (1, 0) if use_1st_coupon_for_neutr_strike else (2, 1)

        den = np.dot(self.__delta_pills[delta_pills_ind:], self.__discounted_pills[index:])
        
        cond_exp = [self.underlying.conditional_expectation(t, 0.)[self.underlying_index, 0] for t in self.__pills[index:]]
        first_part = np.multiply(self.__discounted_pills[index:], cond_exp)
        
        num = np.dot(self.__delta_pills[delta_pills_ind:], first_part)
        
        return num / den
        
    def price(self, t, coupon=False):
        fst_payment_idx = np.searchsorted(self.__pills, t, side='right')
        if fst_payment_idx >= len(self.__pills):
            return 0.
        
        df_t = self.discount_factor(t)
        
        first_coupon = 0.
        if coupon:
            t_i_star_m_1 = self.__pills[fst_payment_idx-1]
            st_i_star_m_1 = self.S(t_i_star_m_1)
        
            first_coupon = self.__delta_pills[fst_payment_idx-1]
            first_coupon *= self.__discounted_pills[fst_payment_idx]
            first_coupon *= st_i_star_m_1 - self.strike
        
        deltas = self.__delta_pills[fst_payment_idx:]

        strike_term = self.__k*np.dot(deltas, self.__discounted_pills[fst_payment_idx + 1:])

        cond_exp = [self.underlying.conditional_expectation(t_i, t)[self.underlying_index, 0]
                    for t_i in self.__pills[fst_payment_idx + 1:]]
        prod_mart_discounted = np.multiply(cond_exp, self.__discounted_pills[fst_payment_idx + 1:])

        spot_term = np.dot(prod_mart_discounted, deltas)

        return (first_coupon + spot_term - strike_term) / df_t

    def __str__(self):
        pill = ("{" + ', '.join(['%.2f']*len(self.pillars))+"}")%tuple(self.pillars)
        return "Swap contract of maturity T = %d years, over S^%d with strike K = %.3f, paying at %s" \
               %(self.maturity, self.underlying_index, self.strike, pill)

    def coupon(self, t):
        if t not in self.__pills:
            return 0.

        index = np.searchsorted(self.__pills, t)

        coupon = self.S(self.__pills[index - 1]) - self.strike
        coupon *= self.__delta_pills[index - 1]

        return coupon

    @classmethod
    def generate_payment_dates(cls, first_date, maturity, step):
        res = np.arange(first_date, maturity+step, step)
        if res[-1] > maturity:
            res = np.delete(res, -1)
            
        return res

EuropeanContract.register(SwapContract)