# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 10:34:55 2015

@author: Yann
"""

import numpy as np

class Accounts(object):
    def __init__(self, members_states):        
        self._states_ = members_states
        self._size_ = self._states_.size        
        self._amounts_ = np.zeros(self._size_)
                
    def put_amount(self, index, amount):
        assert (0 <= index and index < self._size_), "index out of bounds"
        if self._states_.is_alive(index):
            self._amounts_[index] = amount
        
    def get_amount(self, index):
        assert (0 <= index and index < self._size_), "index out of bounds"        
        return self._amounts_[index]
    
    @property
    def amounts(self):
        return self._amounts_
    
    @property
    def states(self):
        return self._states_
    
    def reset(self, initial_count=0.):
        self._states_.resurrect_all()
        self._amounts_ = np.zeros(self._size_)
        if initial_count != 0.:
            self._amounts_.fill(initial_count)
    
#####################################################################
        
class DFAccounts(Accounts):
    def __init__(self, members_states):
        super(DFAccounts, self).__init__(members_states)
    
    def total_default_fund(self, only_surviving=True):
        to_substract = 0
        if only_surviving:
            dead_indexes = [i for (i, s) in enumerate(self._states_.alive_states)  if not s]
            to_substract = np.sum(self._amounts_[dead_indexes])
            
        return np.sum(self._amounts_) - to_substract
    
    def surviving_default_fund(self):
        return np.array([self._amounts_[i] if self._states_.is_alive(i) else 0 for i in range(self._size_)])
    
    def mean_contribution(self):        
        alive_number = np.count_nonzero(self._states_.alive_states)
        return np.sum(self.surviving_default_fund()) / alive_number