from .european import EuropeanContract, ContractType


class FutureContract(EuropeanContract):
    def __init__(self, underlying, df_process, maturity, underlying_index=0):
        super(FutureContract, self).__init__(underlying, maturity, df_process, underlying_index)
        self.__df_mat = self.discount_factor(self.maturity)
        
    def price(self, t):
        if t > self.maturity:
            return 0.

        st = self.S(t)
        res = self.discount_factor(t) / self._df_mat_ * st
        return res

    def coupon(self, t):
        raise NotImplementedError("To do")

    @property
    def asset_class(self):
        return ContractType.equity

    def __str__(self):
        return "Future contract of maturity T = %d years over S^%d"%(self.maturity, self.underlying_index)
    
EuropeanContract.register(FutureContract)