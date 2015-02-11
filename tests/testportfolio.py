# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 11:08:29 2015

@author: Yann
"""

def test_portfolio_construction():
    import numpy as np    
    
    from maths.montecarlo.constantpath import ConstantPath
    
    from finance.products.european.future import FutureContract    
    from finance.products.european.forward import ForwardContract
    from finance.discountfactor import ConstantRateDiscountFactor  

    from ccp.regulation.exposure import *
        
    from ccp.portfolio import CCPPortfolio
    
    time_grid = ConstantPath.generate_time_grid(0, 2, 0.01)
    x0 = [0.3, 0.35, 0.4]    
    process = ConstantPath(x0, time_grid)
    
    df = ConstantRateDiscountFactor(0.03)    

    # Futures    
    
    T1 = np.argmax(time_grid>=1.2)
    T1 = time_grid[T1]
    future = FutureContract(process, df, T1)
    pfuture0 = future.price(0)
    exposure_future = FutureQuantileExposure(1, 0, 1, 0.99, df)
    
    # Forwards
    
    T2 = 1.6
    K = 0.1
    forward = ForwardContract(process, df, T2, K, 1)
    pforward0 = forward.price(0)    
    exposure_forward = ForwardQuantileExposure(1, 0, 1, 0.99, df)    
    
    positions = [[10, 5], [5, -3], [-15, -2]]
    
    derivatives = [future, forward]
    prices = [pfuture0, pforward0]
    exposures = [exposure_future, exposure_forward]
        
    port = CCPPortfolio(positions, derivatives, prices, exposures)
    
#    print port.derivatives
#    print port.amounts
#    print port.weights[:, 0]
#    print port.cm_number  
    
    print port.compute_exposure(1)
    print port.compute_exposure(time_grid[1])
    
    import matplotlib.pyplot as plt
    
    kw = {'risk_period': 1./12}
    expM = [port.compute_exposure(t, **kw)[0, 0] for t in time_grid]
    plt.plot(time_grid, expM, label='Month')
    
    kw2 = {'risk_period': 5./360}
    expD = [port.compute_exposure(t, **kw2)[0, 0] for t in time_grid]
    plt.plot(time_grid, expD, label='Day')
    
    ratio = [d/m for d, m in zip(expD, expM)]
    plt.plot(time_grid, ratio, label='Ratio')    
    
    plt.legend(loc='best')
    
    
    
    #plt.plot(time_grid, [port.compute_exposure(t)[1, 0] for t in time_grid])
    #plt.plot(time_grid, [port.compute_exposure(t)[2, 0] for t in time_grid])
    
    
    