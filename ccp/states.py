# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 10:34:11 2015

@author: Yann
"""

import numpy as np


class MembersState(object):
    def __init__(self, cm_number):
        self.__size = cm_number
        self.__accounts = []
        self.resurrect_all()
        
    def die(self, *indexes):
        self.__surviving_people[list(indexes)] = False
    
    def is_alive(self, index):
        return self.__surviving_people[index]
        
    @property
    def alive_states(self):
        return self.__surviving_people
    
    def resurrect_all(self):
        self.__surviving_people = np.ones(self.__size, dtype=bool)
        for acc in self.__accounts:
            acc.reset()

    def _add_account(self, account):
        if account not in self.__accounts:
            self.__accounts.append(account)

    def _remove_account(self, account):
        if account in self.__accounts:
            del self.__accounts[account]

    @property
    def size(self):
        return self.__size