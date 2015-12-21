import numpy
from scipy.stats import norm

from finance.portfolio import CCPPortfolio, CSAPortfolio


class CSARegulatoryCapital(object):
    """
    RWA based on http://www.bis.org/bcbs/irbriskweight.pdf
    KCVA based on http://papers.ssrn.com/sol3/papers.cfm?abstract_id=2400324
    """

    __df_prob_ratings_in_pc = [{"AAA": 0}, {"AA": 0.02}, {"A": 0.06},
                               {"BBB": 0.17}, {"BB": 1.06}, {"B": 3.71},
                               {"CCC": 12.81}, {"CC": 38.0}]

    __ratings_weights_kcva_in_pc = {"AAA": 0.7, "AA": 0.7, "A": 0.8,
                                    "BBB": 1.0, "BB": 2.0, "B": 3.0,
                                    "CCC": 10.0, "CC": 15.0}
    __systemic_factor = norm.ppf(0.999)
    __min_pd = 0.0003

    @classmethod
    def __b(cls, x):
        return (0.11852 - 0.05478 * numpy.log(x)) ** 2

    @classmethod
    def __compute_correl(cls, default_proba):
        one_minus_exp_m50 = 1. - numpy.exp(-50)
        one_minus_exp_m50_pd = 1. - numpy.exp(-50 * default_proba)

        term1 = 0.12 * one_minus_exp_m50_pd / one_minus_exp_m50
        term2 = 0.24 * (1. - one_minus_exp_m50_pd) / one_minus_exp_m50

        return term1 + term2

    def __init__(self, ead_handler, states, df_prob_models, recoveries, cap_ratio=0.08):
        if not isinstance(ead_handler.portfolio, CSAPortfolio):
            raise ValueError("The EAD portfolio is not a CSAPortfolio")
        
        self.__ead = ead_handler
        self.__states = states
        self.__default_probs = df_prob_models
        self.__recoveries = recoveries

        self.__cap_ratio = cap_ratio

    @property
    def portfolio(self):
        return self.__ead.portfolio

    def __compute_kcva_weight(self, cp_index):
        rating = self.get_rating(cp_index)
        weight = self.__ratings_weights_kcva_in_pc[rating]

        return weight / 100.0

    def __compute_effective_mat(self, cp_index, t):
        positions = self.portfolio.counterparties_positions_from_self[cp_index, :]
        notionals = numpy.multiply(positions, self.portfolio.notionals)

        notional = notionals.sum()

        ttm = [d.maturity - t for d in self.portfolio.derivatives]
        not_weighted_avg = numpy.dot(ttm, notionals) / notional

        return not_weighted_avg

    def get_rating(self, cp_index):
        df_prob = 100.0 * self.__default_probs[cp_index].default_proba(1.0)

        rating = None
        for d in self.__df_prob_ratings_in_pc:
            for k, v in d.iteritems():
                if df_prob > v:
                    rating = k
                    break

        if not rating:
            rating = self.__df_prob_ratings_in_pc[-1].keys()[0]

        return rating

    def compute_k(self, cp_index, t):
        r = self.__recoveries[cp_index]
        lgd = 1.-r

        dp = self.__default_probs[cp_index].default_proba(1.)
        dp = numpy.maximum(dp, self.__min_pd)

        correl = self.__compute_correl(dp)

        tmp = norm.ppf(dp) + numpy.sqrt(correl) * self.__systemic_factor
        gauss_factor = norm.cdf(tmp / numpy.sqrt(1. - correl))

        M_unfloored_uncapped = self.__compute_effective_mat(cp_index, t)
        M = numpy.minimum(5., numpy.maximum(1., M_unfloored_uncapped))

        b_dp = self.__b(dp)

        coeff = (1. + (M - 2.5) * b_dp) / (1. - 1.5 * b_dp)

        return lgd * (gauss_factor - dp) * coeff

    def compute_rwa(self, cp_index, t, **kwargs):
        tmp = 12.5 * self.compute_k(cp_index, t) 
        tmp *= self.compute_eads(cp_index, t, **kwargs)
        return tmp

    def compute_eads(self, cp_index, t, **kwargs):
        kwargs2 = kwargs.copy()
        kwargs2['t'] = t

        positions = self.portfolio.counterparties_positions_from_self[cp_index]
        positions = positions.reshape((1, len(positions)))
        kwargs2['positions'] = positions

        return self.__ead(**kwargs2)

    def compute_k_ccr(self, cp_index, t, **kwargs):
        if not self.__states.is_alive(cp_index):
            return 0.

        rwa = self.compute_rwa(cp_index, t, **kwargs).sum()
        return self.__cap_ratio * rwa

    def compute_k_cva(self, cp_index, t, **kwargs):
        if not self.__states.is_alive(cp_index):
            return 0.

        w_i = self.__compute_kcva_weight(cp_index)
        m_i = self.__compute_effective_mat(cp_index, t)

        if m_i == 0.:
            return 0.

        ead_i = self.compute_eads(cp_index, t, **kwargs).sum()
        discounted_ead_i = (1. - numpy.exp(-0.05*m_i)) / (0.05 * m_i) * ead_i

        return 2.33 * 0.5 * w_i * m_i * discounted_ead_i


class CCPRegulatoryCapital2014(object):
    """
    Implemented based on http://www.bis.org/publ/bcbs282.pdf
    """

    def __init__(self, ead_handler, df_account, sig, cap_ratio=0.08, risk_weight=0.2):
        if not isinstance(ead_handler.portfolio, CCPPortfolio):
            raise ValueError("The EAD portfolio is not a CCPPortfolio")
        
        self.__ead = ead_handler
        self.__df_account = df_account
        self.__sig = sig
        self.__states = df_account.states
        self.__rw = risk_weight
        self.__cap_ratio = cap_ratio

    @property
    def portfolio(self):
        return self.__ead.portfolio

    @property
    def capital_ratio(self):
        return self.__cap_ratio

    @property
    def risk_weight(self):
        return self.__rw

    def compute_eads(self, t, **kwargs):
        kwargs2 = kwargs.copy()
        kwargs2['t'] = t
        return self.__ead(**kwargs2)

    def compute_k_ccp(self, t, **kwargs):
        eads = self.compute_eads(t, **kwargs)
        alive_eads = eads[self.__states.alive_states]

        return self.__rw * self.__cap_ratio * alive_eads.sum()

    def compute_k_cm(self, cm_index, t, **kwargs):
        total_df = self.__df_account.total_default_fund().sum() + self.__sig.value

        if total_df <= 0:
            # If we are here, it means that the default fund is zero
            # for everyone and that the SIG is zero.
            return 0.

        k_ccp = self.compute_k_ccp(t, **kwargs)

        df = self.__df_account.get_amount(cm_index).sum()
        firt_term = k_ccp * (df / total_df)
        second_term = 0.08*0.02*df

        return numpy.maximum(firt_term, second_term)
