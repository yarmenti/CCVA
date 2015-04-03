__author__ = 'Yann'


import numpy as np

class Accounts(object):
    def __init__(self, states, derivatives_nb):
        self.states = states
        self.__amounts = np.zeros([self.__states.size, derivatives_nb])

    @property
    def states(self):
        return self.__states

    @states.setter
    def states(self, value):
        try:
            self.__states._remove_account(self)
        except:
            pass

        self.__states = value
        self.__states._add_account(self)

    @property
    def amounts(self):
        return np.array(self.__amounts, copy=True)

    def put_amounts(self, index, amounts):
        if self.__states.is_alive(index):
            self.__amounts[index, :] = amounts

    def get_amount(self, index):
        return self.__amounts[index, :]

    def reset(self, **kwargs):
        self.__amounts = np.zeros(self.__amounts.shape)


class DFAccounts(Accounts):
    def __filtered_default_fund(self, only_surviving):
        indexes = np.arange(self.amounts.shape[1] - 1)
        if only_surviving:
            indexes = indexes[self.states.alive_states]

        return self.amounts[indexes, :]

    def total_default_fund(self, only_surviving=True):
        res = self.__filtered_default_fund(only_surviving)
        return res.sum(axis=0)

    def mean_contribution(self):
        total = self.total_default_fund()
        alive_number = np.count_nonzero(self.states.alive_states)

        return total/alive_number




        #mean_contrib =