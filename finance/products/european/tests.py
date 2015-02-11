# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 15:53:27 2015

@author: Yann
"""

import numpy as np

def test_future():
    import matplotlib.pyplot as plt    
    
    from maths.montecarlo.constantpath import ConstantPath
    from future import FutureContract
    from finance.discountfactor import ConstantRateDiscountFactor

    time_grid = ConstantPath.generate_time_grid(0, 2, 0.01)    
    
    S = ConstantPath([10., 9.], time_grid)    
    
    df = ConstantRateDiscountFactor(0.03)
    future = FutureContract(S, df, 1.5)  
    
    plt.plot(time_grid, [future.price(t) for t in time_grid])
    plt.show()
    
def test_forward():
    import matplotlib.pyplot as plt    
    
    from maths.montecarlo.constantpath import ConstantPath
    from forward import ForwardContract
    from finance.discountfactor import ConstantRateDiscountFactor

    time_grid = ConstantPath.generate_time_grid(0, 2, 0.01)    
    
    S = ConstantPath([10., 9.], time_grid)    
    
    df = ConstantRateDiscountFactor(0.03)
    forward = ForwardContract(S, df, 1.5, 10.)  
    
    plt.plot(time_grid, [forward.price(t) for t in time_grid])
    plt.show()

def test_swap():
    import matplotlib.pyplot as plt    
    
    from maths.montecarlo.constantpath import ConstantPath
    from swap import SwapContract
    from finance.discountfactor import ConstantRateDiscountFactor

    time_grid = ConstantPath.generate_time_grid(0, 2, 0.01)    
    
    S = ConstantPath([10., 9.], time_grid)    
    
    df = ConstantRateDiscountFactor(0.03)
    pillars = SwapContract.generate_payment_dates(0, 2, .25)
    
    swap = SwapContract(S, df, pillars)  
    prices = [swap.price(t) for t in time_grid]
#    print prices
    plt.plot(time_grid, prices)    
    plt.show()
    
def quantile_future():
    from maths.montecarlo.constantpath import ConstantPath
    from future import FutureContract, ind_future_brownian_quantile
    from finance.discountfactor import ConstantRateDiscountFactor
    
    time_grid = ConstantPath.generate_time_grid(0, 2, 0.01)        
    S = ConstantPath([100.], time_grid)        
    df = ConstantRateDiscountFactor(0.0)
    
    future = FutureContract(S, df, 1.5)      
        
    print ind_future_brownian_quantile(0, 1., 0., 1., 0.99, future, 1., df)    
    from scipy.stats import norm
    print norm.ppf(0.99)
    
def quantile_forward():
    from maths.montecarlo.constantpath import ConstantPath
    from forward import ForwardContract, ind_forward_brownian_quantile
    from finance.discountfactor import ConstantRateDiscountFactor
    
    time_grid = ConstantPath.generate_time_grid(0, 2, 0.01)        
    S = ConstantPath([100.], time_grid)        
    df = ConstantRateDiscountFactor(0.0)
    
    forward = ForwardContract(S, df, 1.5, 1)      
        
    print ind_forward_brownian_quantile(0, 1., 0., 1., 0.99, forward, 1., df)    
    from scipy.stats import norm
    print norm.ppf(0.99)