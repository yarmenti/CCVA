from .european import EuropeanContract, ContractType


class ForwardContract(EuropeanContract):
    def __init__(self, underlying, df_process, maturity, strike, underlying_index=0):
        super(ForwardContract, self).__init__(underlying, maturity, df_process, underlying_index)        
        self.__df_mat = self.discount_factor(self.maturity)
        
        self.__k = strike
        
    def price(self, t):
        if t > self.maturity:
            return 0.

        st = self.S(t)
        res = st - self.__df_mat/self.discount_factor(t) * self.__k
        return res
        
    @property
    def strike(self):
        return self.__k

    @property
    def asset_class(self):
        return ContractType.equity

    def coupon(self, t):
        raise NotImplementedError("To do")

    def __str__(self):
        return "Forward contract of strike K = %d and maturity T = %d years over S^%d"\
               %(self.strike, self.maturity, self.underlying_index)
    
EuropeanContract.register(ForwardContract)