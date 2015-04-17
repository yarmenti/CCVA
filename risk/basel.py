
import numpy as np
from scipy.stats import norm


class RegulatoryCapital(object):
    __systemic_factor = norm.ppf(0.999)
    __min_pd = 0.0003
    __beta_exposure = 1.4
    __df_prob_ratings_in_pc = [{"AAA": 0}, {"AA": 0.02}, {"A": 0.06},
                               {"BBB": 0.17}, {"BB": 1.06}, {"B": 3.71},
                               {"CCC": 12.81}, {"CC": 38.0}]

    __ratings_weights_kcva_in_pc = {"AAA": 0.7, "AA": 0.7, "A": 0.8,
                                    "BBB": 1.0, "BB": 2.0, "B": 3.0,
                                    "CCC": 10.0, "CC": 15.0}

    def __init__(self, vm_accounts, im_accounts, portfolio, default_proba, recoveries, bank_index=0, cap_ratio=0.08):
        self.__cap_ratio = cap_ratio
        self.__bank_idx = bank_index
        self.__port = portfolio
        self.__im_acc = im_accounts
        self.__vm_acc = vm_accounts

        if self.__port.bank_numbers != len(default_proba):
            raise ValueError("The portfolio.nb_banks (%s) "
                             "!= len(default_proba) (%s)"%(self.__port.bank_numbers, len(default_proba)))
        self.__default_probs = default_proba

        if self.__port.bank_numbers != len(recoveries):
            raise ValueError("The portfolio.nb_banks != len(recoveries)")
        self.__recoveries = recoveries

    @classmethod
    def __b(cls, x):
        return (0.11852 - 0.05478 * np.log(x)) ** 2

    @classmethod
    def __compute_correl(cls, default_proba):
        one_minus_exp_m50 = 1. - np.exp(-50)
        one_minus_exp_m50_pd = 1. - np.exp(-50 * default_proba)

        term1 = 0.12 * one_minus_exp_m50_pd / one_minus_exp_m50
        term2 = 0.24 * (1. - one_minus_exp_m50_pd) / one_minus_exp_m50

        return term1 + term2

    def __compute_regulatory_weight(self, counterparty_index, t, risk_horizon):
        r = self.__recoveries[counterparty_index]
        lgd = 1.-r

        dp = self.__default_probs[counterparty_index].default_proba(risk_horizon)
        dp = np.maximum(dp, self.__min_pd)

        correl = self.__compute_correl(dp)

        tmp = norm.ppf(dp) + np.sqrt(correl) * self.__systemic_factor
        gauss_factor = norm.cdf(tmp / np.sqrt(1. - correl))

        M = self.__compute_effective_mat(counterparty_index, t)
        b_dp = self.__b(dp)

        coeff = (1. + (M - 2.5) * b_dp) / (1. - 1.5 * b_dp)

        return lgd * (gauss_factor - dp) * coeff

    def __compute_effective_mat(self, counterparty_index, t):
        projection = self.__port.compute_projection(self.__bank_idx, counterparty_index)
        positions = self.__port.positions[self.__bank_idx, :]

        notionals = np.multiply(projection, positions)
        notional = notionals.sum()

        ttm = [d.maturity - t for d in self.__port.derivatives]
        not_weighted_avg = np.dot(ttm, notionals) / notional

        return np.minimum(5., np.maximum(1., not_weighted_avg))

    def __compute_ead(self, counterparty_index, t, risk_horizon, conf_level, **kwargs):
        exposure = self.__port.compute_exposure(t,
                                                risk_period=risk_horizon,
                                                conf_level=conf_level,
                                                from_=self.__bank_idx,
                                                towards_=counterparty_index,
                                                total=True,
                                                **kwargs
                                                )

        vm = self.__vm_acc.amounts[counterparty_index]
        im = self.__im_acc.amounts[counterparty_index]

        ead = np.maximum(exposure - vm - im, 0)

        return ead

    def __compute_kcva_weight(self, counterparty_index):
        rating = self.get_rating(counterparty_index)
        weight = self.__ratings_weights_kcva_in_pc[rating]

        return weight / 100.0

    def get_rating(self, counterparty_index):
        df_prob = 100.0*self.__default_probs[counterparty_index].default_proba(1.0)

        rating = None
        for d in self.__df_prob_ratings_in_pc:
            for k, v in d.iteritems():
                if df_prob > v:
                    rating = k
                    break

        if not rating:
            rating = self.__df_prob_ratings_in_pc[-1].keys()[0]

        return rating

    def compute_kccr(self, counterparty_index, t, risk_horizon=1., conf_level=0.999, **kwargs):
        w = self.__compute_regulatory_weight(counterparty_index, t, risk_horizon)
        ead = self.__compute_ead(counterparty_index, t, risk_horizon, conf_level, **kwargs)

        ead *= self.__beta_exposure

        return self.__cap_ratio*12.5*ead[0]*w

    def compute_kcva(self, counterparty_index, t, risk_horizon=1., conf_level=0.999, **kwargs):
        mult = 0.5*2.33
        sqrt_h = np.sqrt(risk_horizon)

        w_i = self.__compute_kcva_weight(counterparty_index)
        m_i = self.__compute_effective_mat(counterparty_index, t)

        ead_i = self.__compute_ead(counterparty_index, t, risk_horizon, conf_level, **kwargs)
        discount = (1.-np.exp(-0.05*m_i))/(0.05*m_i)

        return mult*sqrt_h*w_i*m_i*discount*ead_i[0]