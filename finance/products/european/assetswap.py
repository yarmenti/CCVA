import numpy as np
from scipy import interpolate

from .european import EuropeanContract, ContractType


class SwapContract(EuropeanContract):
    def __init__(self, underlying, df_process, dates, underlying_index=0):
        super(SwapContract, self).__init__(underlying, dates[-1], df_process, underlying_index)
        
        self.__pills = np.sort(dates)

        self.__vect_df = np.vectorize(self.discount_factor)
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
        return np.array(self.__pills, copy=True)

    @property
    def asset_class(self):
        return ContractType.interest_rate

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

        #return first_coupon + term1 + term2
        return term1 + term2
      
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