import numpy as np


class CCPRegulatoryCapital(object):
    def __init__(self, beta, im_accounts, df_accounts, sig, portfolio, **kwargs):        
        cm_nb = df_accounts.size
        
        self.__df = df_accounts
        self.__im = im_accounts
        
        self.__eqty = sig
        self.__coeff = 1 + beta*(cm_nb/(cm_nb - 2))
        self.__port = portfolio
        
        self._init_kwargs_(**kwargs)
        
    def _init_kwargs_(self, **kwargs):
        self.__ead_coeff = kwargs.get('ead_coeff', 1.)
        
    def _compute_k_cms_(self, t):
        e = self.__eqty()
        df_tot = self.__df.total_default_fund()
        df_prime_cm = np.maximum(df_tot - 2*self.__df.mean_contribution(), 0)
        df_prime = e + df_prime_cm
        
        k_ccp = self.compute_k_ccp(t)
        
        c1 = 0.0016
        if df_prime > 0:
            c1 = np.maximum(0.0016, 0.016*(k_ccp/df_prime)**0.3)
        
        c2 = 1.0
        mu = 1.2
        
        res = 0
        if df_prime < k_ccp:
            res = c2 * (mu*(k_ccp-df_prime) + df_prime_cm)
        elif e < k_ccp and k_ccp < df_prime:
            res = c2*(k_ccp-e) + c1*(df_prime-k_ccp)
        elif k_ccp <= e:
            res = c1*df_prime_cm
            
        return res        
        
    def compute_k_cm(self, index, t):
        total_df = self.__df.total_default_fund()
        if total_df <= 0:
            raise RuntimeError("The total default fund must be > 0")
        
        k_cms = self._compute_k_cms_(t)
        
        df = self.__df.get_amount(index)
        
        return self.__coeff * df/total_df * k_cms
    
    def compute_k_ccp(self, t):
        capital_ratio = 0.08
        risk_weight = 0.2
        
        ebrms = self.__port.compute_exposure(t)
                
        ims = self.__im.amounts
        dfs = self.__df.amounts
        states = self.__im.states.alive_states
        
        res = 0
        for ebrm, im, df, is_alive in zip (ebrms.flat, ims, dfs, states):            
            if is_alive:
                mod_ebrm = ebrm * self.__ead_coeff
                res += np.maximum(mod_ebrm - im - df, 0)
        
        res *= capital_ratio * risk_weight
        
        return res


class NoDFRegulatoryCapital(CCPRegulatoryCapital):
    def compute_k_cm(self, index, t):
        return 0.