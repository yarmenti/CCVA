import numpy as np
from pandas.tslib import _Timedelta
from scipy import interpolate

from .european import EuropeanContract, ContractType


class SwapContract(EuropeanContract):
    def __init__(self, underlying, df_process, dates, underlying_index=0):
        super(SwapContract, self).__init__(underlying, dates[-1], df_process, underlying_index)
        
        self.__pills = np.sort(dates)

        self.__vect_df = np.vectorize(self.discount_factor)
        self.__discounted_pills = self.__vect_df(self.__pills)
        self.__delta_pills = np.ediff1d(self.__pills)

        self.__k = self.compute_neutral_strike()

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

    def compute_neutral_strike(self):
        discounts = self.__discounted_pills[1:]
        den = np.dot(self.__delta_pills, discounts)

        cond_expect = cond_expect = lambda x: self.udl_cond_expect(x, 0.)
        cond_exp = map(cond_expect, self.__pills[:-1])

        num = np.multiply(self.__delta_pills, cond_exp)
        num = np.dot(num, discounts)

        return num / den

    def price(self, t, incl_next_coupon=False):
        next_pymt_idx = np.searchsorted(self.__pills, t, side='right')
        if next_pymt_idx >= len(self.__pills):
            return 0.
        
        df_t = self.discount_factor(t)
        first_coupon = 0.
        if incl_next_coupon:
            last_coupon_date = self.__pills[next_pymt_idx - 1]
            st = self.S(last_coupon_date)
        
            first_coupon = self.__delta_pills[next_pymt_idx - 1]
            first_coupon *= self.__discounted_pills[next_pymt_idx]
            first_coupon *= st - self.strike

        deltas = self.__delta_pills[next_pymt_idx:]

        strike_term = self.__k * np.dot(deltas, self.__discounted_pills[next_pymt_idx + 1:])
        cond_expect = lambda x: self.udl_cond_expect(x, t)
        
        cond_exp = map(cond_expect, self.__pills[next_pymt_idx:-1])

        prod_mart_discounted = np.multiply(cond_exp, self.__discounted_pills[next_pymt_idx + 1:])
        spot_term = np.dot(prod_mart_discounted, deltas)

        return (first_coupon + spot_term - strike_term) / df_t

    def udl_cond_expect(self, T, t):
    	return self.underlying.conditional_expectation(T, t)[self.underlying_index, 0]

    def __str__(self):
        pill = ("{" + ', '.join(['%.2f'] * len(self.pillars)) + "}") % tuple(self.pillars)
        return "Swap contract of maturity T = %d years, over S^%d with strike K = %.3f, paying at %s" \
               % (self.maturity, self.underlying_index, self.strike, pill)

    def coupon(self, t):
        if t not in self.__pills:
            return 0.

        index = np.searchsorted(self.__pills, t)

        coupon = self.S(self.__pills[index - 1]) - self.strike
        coupon *= self.__delta_pills[index - 1]

        return coupon

    @classmethod
    def generate_payment_dates(cls, first_date, maturity, step):
        res = np.arange(first_date, maturity + step, step)
        if res[-1] > maturity:
            res = np.delete(res, -1)
            
        return res


EuropeanContract.register(SwapContract)