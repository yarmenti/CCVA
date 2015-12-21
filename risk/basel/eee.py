import numpy as np
from scipy.stats import norm

from maths.montecarlo.processes.brownianmotions import GeometricBrownianMotion
from finance.products.european.swap import SwapContract


class BlackScholesSwapVaREffExpectExposure(object):
    def __init__(self, swap, index=0):
        if not isinstance(swap, SwapContract):
            raise ValueError("swap is not of type SwapContract")

        self.__swap = swap

        if not isinstance(self.__swap.underlying, GeometricBrownianMotion):
            raise ValueError("underlying is not of type Process")

        self.__udl = self.__swap.underlying
        self.__idx = index

        self.__df = self.__swap.discount_factor

        self.__drift = self.__udl.drifts[index, 0]
        self.__vol = self.__udl.vols[index, 0]
        
    def __call__(self, v, positions, **kwargs):
        t = kwargs['t']
        St = self.__udl(t)[self.__idx]

        risk_period = kwargs['risk_period']
        v_p_delta = v + risk_period

        inv_df = 1. / self.__df(v_p_delta)
        exp_factor = np.exp(-self.__drift * t - 0.5 * (self.__vol**2) * risk_period)

        swap_pillars = self.__swap.pillars
        h = self.__swap.delta_time
        nxt_coupon_idx = np.searchsorted(swap_pillars, v_p_delta, side='right')

        sum_ = 0.
        for ii in xrange(nxt_coupon_idx, len(swap_pillars)):
            sum_ += h[ii - 1] * self.__df(swap_pillars[ii]) * np.exp(self.__drift * swap_pillars[ii-1])

        partial_res = St * inv_df * exp_factor * sum_
        
        vol = self.__vol * np.sqrt(risk_period)

        res = np.empty(2)
        
        alpha = kwargs["alpha"]
        if alpha < 0.5:
            alpha = 1. - alpha
  
        gauss_quantile = norm.ppf(alpha)
        var_up = gauss_quantile
        es_up = norm.pdf(var_up) / (1.-alpha)

        res[0] = (1.-alpha) * partial_res * ( np.exp(vol * es_up) - np.exp(vol * var_up) )
        res[1] = (1.-alpha) * partial_res * ( np.exp(-vol * es_up) - np.exp(-vol * var_up) )
                
        pos = np.tile(positions, 2)
        prod = np.multiply(res, pos)
        
        return np.reshape(np.amax(prod, axis=1), positions.shape)
        