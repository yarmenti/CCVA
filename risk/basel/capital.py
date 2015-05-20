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

    def __init__(self, vm_accounts, im_accounts, portfolio, default_proba, recoveries, exposures, bank_index=0, cap_ratio=0.08):
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

        if len(self.__port.derivatives) != len(exposures):
            raise ValueError("The len(portfolio.derivatives) != len(exposures)")
        self.__exposures = exposures

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
        projection = self.portfolio.compute_projection(self.__bank_idx, counterparty_index)
        positions = self.portfolio.positions[self.__bank_idx, :]

        notionals = np.multiply(projection, positions)
        notional = notionals.sum()

        ttm = [d.maturity - t for d in self.__port.derivatives]
        not_weighted_avg = np.dot(ttm, notionals) / notional

        return np.minimum(5., np.maximum(1., not_weighted_avg))

    def _compute_potential_future_loss(self, t, risk_horizon, conf_level, **kwargs):
        directions = [1, -1]
        losses = np.zeros(self.portfolio.positions.shape)
        for (ii, e) in enumerate(self.exposures):
            exposure = e(t=t, risk_horizon=risk_horizon,
                         conf_level=conf_level, **kwargs)

            for d, exp in zip(directions, exposure):
                temp = self.portfolio.directions[:, ii] == d
                losses[:, ii][temp] = np.maximum(self.portfolio.positions[:, ii][temp] * exp, 0)

        return losses

    # def __compute_ead(self, counterparty_index, t, risk_horizon, conf_level, **kwargs):
    #     exposure = self.__port.compute_exposure(t,
    #                                             risk_period=risk_horizon,
    #                                             conf_level=conf_level,
    #                                             from_=self.__bank_idx,
    #                                             towards_=counterparty_index,
    #                                             total=True,
    #                                             **kwargs
    #                                             )
    #
    #     vm = self.__vm_acc.amounts[counterparty_index]
    #     im = self.__im_acc.amounts[counterparty_index]
    #
    #     ead = np.maximum(exposure - vm - im, 0)
    #
    #     return ead

    def __compute_ead(self, counterparty_index, t, risk_horizon, conf_level, **kwargs):
        projection = self.__port.compute_projection(self.__bank_idx, counterparty_index)

        losses = kwargs["pf_losses"] if "pf_losses" in kwargs else self._compute_potential_future_loss(t, risk_horizon, conf_level, **kwargs)
        losses = losses[self.__bank_idx, :]
        losses = np.multiply(losses, projection)

        loss = losses.sum()
        vm = self.__vm_acc.amounts[self.__bank_idx].sum()
        im = self.__im_acc.amounts[self.__bank_idx].sum()

        ead = np.maximum(loss - vm - im, 0)

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

        return self.__cap_ratio*12.5*ead*w

    def compute_kcva(self, counterparty_index, t, risk_horizon=1., conf_level=0.999, **kwargs):
        mult = 0.5*2.33
        sqrt_h = np.sqrt(risk_horizon)

        w_i = self.__compute_kcva_weight(counterparty_index)
        m_i = self.__compute_effective_mat(counterparty_index, t)

        ead_i = self.__compute_ead(counterparty_index, t, risk_horizon, conf_level, **kwargs)
        discount = (1.-np.exp(-0.05*m_i))/(0.05*m_i)

        return mult*sqrt_h*w_i*m_i*discount*ead_i

    @property
    def portfolio(self):
        return self.__port

    @property
    def exposures(self):
        return self.__exposures


class CCPRegulatoryCapital2012(RegulatoryCapital):
    capital_ratio = 0.08

    def __init__(self, vm_accounts, im_accounts, df_accounts, sig, beta, portfolio, exposures, risk_weight=0.2):
        cm_nb = df_accounts.size

        self.__vm_acc = vm_accounts
        self.__im_acc = im_accounts
        self.__df_acc = df_accounts

        self.__sig = sig
        self.__coeff = 1 + (beta * cm_nb / (cm_nb - 2))
        self.__port = portfolio

        if len(exposures) != len(self.__port.derivatives):
            raise ValueError("The exposures do not have the same length (%s) "
                             "as the nb of derivatives (%s)"%(len(exposures), len(self.__port.derivatives)))

        self.__exposures = exposures

        self.__risk_weight = risk_weight

    # def __compute_eads(self, t, risk_horizon, conf_level, **kwargs):
    #     exposures = self.__port.compute_exposure(t, risk_period=risk_horizon, conf_level=conf_level, **kwargs)
    #
    #     agg_exposures = np.sum(exposures, axis=1)
    #
    #     vm = np.sum(self.__vm_acc.amounts, axis=1)
    #     im = np.sum(self.__im_acc.amounts, axis=1)
    #     df = np.sum(self.__df_acc.amounts, axis=1)
    #
    #     zeros = np.zeros(agg_exposures.shape)
    #
    #     return np.maximum(agg_exposures - vm - im - df, zeros)

    def __compute_eads(self, t, risk_horizon, conf_level, **kwargs):
        losses = kwargs["pf_losses"] if "pf_losses" in kwargs else self._compute_potential_future_loss(t, risk_horizon, conf_level, **kwargs)

        agg_losses = losses.sum(axis=1)

        vm = np.sum(self.__vm_acc.amounts, axis=1)
        im = np.sum(self.__im_acc.amounts, axis=1)
        df = np.sum(self.__df_acc.amounts, axis=1)

        zeros = np.zeros(agg_losses.shape)

        return np.maximum(agg_losses - vm - im - df, zeros)

    def __compute_kcms(self, k_ccp):
        e = self.__sig.value
        df_tot = self.__df_acc.total_default_fund().sum()
        df_prime_cm = np.maximum(df_tot - 2*self.__df_acc.mean_contribution().sum(), 0)
        df_prime = e + df_prime_cm

        c1 = 0.0016
        if df_prime > 0:
            c1 = np.maximum(0.0016, 0.016*(k_ccp/df_prime)**0.3)

        c2 = 1.0
        mu = 1.2

        res = 0.
        if df_prime < k_ccp:
            res = c2 * (mu*(k_ccp-df_prime) + df_prime_cm)
        elif e < k_ccp < df_prime:
            res = c2*(k_ccp-e) + c1*(df_prime-k_ccp)
        elif k_ccp <= e:
            res = c1*df_prime_cm

        return res

    def compute_k_ccp(self, t, risk_horizon, conf_level, **kwargs):
        eads = self.__compute_eads(t, risk_horizon, conf_level, **kwargs)

        states = self.__im_acc.states
        alive_eads = eads[states.alive_states]

        return self.__risk_weight * self.capital_ratio * alive_eads.sum()

    def compute_kcm(self, clearing_member_index, t, risk_horizon=1., conf_level=0.999, **kwargs):
        total_df = self.__df_acc.total_default_fund().sum()
        if total_df <= 0:
            raise RuntimeError("The total default fund must be > 0")

        df = self.__df_acc.get_amount(clearing_member_index).sum()
        ratio_df = df / total_df

        k_ccp = self.compute_k_ccp(t, risk_horizon, conf_level, **kwargs)
        k_cms = self.__compute_kcms(k_ccp)

        return self.__coeff * ratio_df * k_cms

    @property
    def portfolio(self):
        return self.__port

    @property
    def exposures(self):
        return self.__exposures

    @property
    def df_account(self):
        return self.__df_acc

    @property
    def sig(self):
        return self.__sig


class CCPRegulatoryCapital2014(CCPRegulatoryCapital2012):
    def __init__(self, vm_accounts, im_accounts, df_accounts, sig, portfolio, exposures, risk_weight=0.2):
        super(CCPRegulatoryCapital2014, self).__init__(vm_accounts, im_accounts, df_accounts, sig, 0., portfolio, exposures, risk_weight)

    def compute_kcm(self, clearing_member_index, t, risk_horizon=1., conf_level=0.999, **kwargs):
        total_df = self.df_account.total_default_fund().sum() + self.sig.value
        if total_df <= 0:
            raise RuntimeError("The total default fund must be > 0")

        df = self.df_account.get_amount(clearing_member_index).sum()

        k_ccp = self.compute_k_ccp(t, risk_horizon, conf_level, **kwargs)

        firt_term = k_ccp * (df/total_df)
        second_term = 0.08*0.02*df

        return np.maximum(firt_term, second_term)