from abc import ABCMeta, abstractmethod

from scipy.stats import norm
import numpy as np
from maths.montecarlo.processes.brownianmotions import BrownianMotion, GeometricBrownianMotion

from maths.montecarlo.processes.historical import HistoricalProcess


class EuropeanAbsExposure(object):
    __metaclass__ = ABCMeta

    def __init__(self, european_contract, df, drift, vol):
        if isinstance(vol, (list, np.ndarray)):
            if len(vol) != 1:
                raise ValueError("Vol must be a scalar")
            vol = vol[0]
        self.__vol = vol

        if isinstance(drift, (list, np.ndarray)):
            if len(drift) != 1:
                raise ValueError("Drift must be a scalar")
            drift = drift[0]
        self.__drift = drift

        self.__contract = european_contract

    def __call__(self, **kwargs):
        t = kwargs['t']
        risk_period = kwargs['risk_period']
        if t + risk_period > self.__contract.maturity:
            risk_period = self.__contract.maturity - t

        alpha = kwargs['conf_level']
        conf_level = [alpha, 1.-alpha]

        drift = kwargs.get('drift', self.__drift)
        vol = kwargs.get('vol', self.__vol)

        res = np.empty([2])
        for (i_, a) in enumerate(conf_level):
            tmp = self.__compute_pl_price(t, risk_period, a, drift, vol)
            res[i_] = tmp

        return res

    def __compute_pl_price(self, t, risk_period, alpha, drift, vol):
        t_ph = t + risk_period
        quantile_inv = self.compute_quantile(alpha)

        lst_pill_idx = np.searchsorted(self.__contract.pillars, t_ph)
        pillars = np.unique(np.append(self.__contract.pillars[:lst_pill_idx], [t, t_ph]))
        pillars.sort()

        process_values = self._compute_diff(t, pillars, drift, vol, quantile_inv)

        tmp_underlying = self.__contract.underlying
        values = np.tile(process_values, (self.__contract.underlying.dimension, 1))

        self.__contract.underlying = HistoricalProcess(pillars, values)
        result = self.__contract.p_and_l(t, t_ph)
        self.__contract.underlying = tmp_underlying

        #coupon_pym_dates = [p for p in self.__contract.pillars if t < p < t_ph]
        #to_add = 0.
        #for p in coupon_pym_dates:
        #    to_add += self.__discount(p)/self.__discount(t_ph) * self.contract.coupon(p)

        #return result + to_add
        return result

    @abstractmethod
    def compute_quantile(self, alpha):
        #return norm.ppf(alpha)
        pass

    @abstractmethod
    def _compute_diff(self, t, pillars, drift, vol, quantile_inv):
        pass

    @property
    def contract(self):
        return self.__contract

    @property
    def discount(self):
        return self.__discount
    
    @property
    def param_names(self):
        return ("conf_level", "risk_period")


class EuropeanVaRExposure(EuropeanAbsExposure):
    def compute_quantile(self, alpha):
        return norm.ppf(alpha)

        
class EuropeanESExposure(EuropeanAbsExposure):
    def compute_quantile(self, alpha):
        return norm.pdf(norm.ppf(alpha)) / alpha


class EuropeanBrownianExposure(EuropeanAbsExposure):
    def _compute_diff(self, t, pillars, drift, vol, quantile_inv):
        process_values = []
        for t_ in pillars:
            if t_ <= t:
                St_ = self.contract.S(t_)
            else:
                if t_ in self.contract.pillars:
                    St_ = self.contract.udl_cond_expect(t_, t__)
                else:
                    St_ = BrownianMotion.compute_next_value(St_, drift, vol, t_-t__, quantile_inv)

            process_values.append(St_)
            t__ = t_

        return process_values


class EuropeanGeomBrownianExposure(EuropeanAbsExposure):
    def _compute_diff(self, t, pillars, drift, vol, quantile_inv):
        process_values = []
        for t_ in pillars:
            if t_ <= t:
                St_ = self.contract.S(t_)
            else:
                if t_ in self.contract.pillars:
                    St_ = self.contract.udl_cond_expect(t_, t__)
                else:
                    St_ = GeometricBrownianMotion.compute_next_value(St_, drift, vol, t_-t__, quantile_inv)[0, 0]

            process_values.append(St_)
            t__ = t_

        return process_values


class EuropeanVaRGeomBrownianExposure(EuropeanGeomBrownianExposure, EuropeanVaRExposure):
    pass


EuropeanAbsExposure.register(EuropeanVaRGeomBrownianExposure)