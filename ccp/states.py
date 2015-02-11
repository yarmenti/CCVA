# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 10:34:11 2015

@author: Yann
"""

import numpy as np

class MembersState(object):
    def __init__(self, cm_number):
        assert (cm_number > 0), "cm_number must be positive"
        self._size_ = cm_number
        self.resurrect_all()
        
    def die(self, *indexes):
        for index in indexes:            
            assert (index < self._size_), "index out of bounds"
        
        self._survivng_people_[list(indexes)] = False
    
    def is_alive(self, index):
        assert (index < self._size_), "index out of bounds"
        return self._survivng_people_[index]
        
    @property
    def alive_states(self):
        return self._survivng_people_
    
    def resurrect_all(self):
        self._survivng_people_ = np.ones(self._size_, dtype=bool)
        
    @property
    def size(self):
        return self._size_
        
