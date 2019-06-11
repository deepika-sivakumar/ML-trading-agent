# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 20:59:26 2019

@author: Deepika

"""
import ManualStrategy as ms
import StrategyLearner as sl
import datetime as dt 
from util import get_data, plot_data
import pandas as pd
import indicators as ind
import marketsimcode as msim
import numpy as np

def author():
    return 'dsivakumar6' # Georgia Tech username

if __name__=="__main__": 
    # Inputs
    symbol = "JPM"
    sd = dt.datetime(2008,1,1)
    ed = dt.datetime(2009,12,31)
    sv = 100000
    commission=0.00
    impact=0.005

    # Calculate trades using Manual Strategy
    manual_strategy = ms.ManualStrategy()
    ms_trades_temp = manual_strategy.testPolicy(symbol = symbol, sd=sd, ed=ed, sv=sv) 
    # Create the manual strategy trades dataframe to contain all the dates
    syms=[symbol]
    dates = pd.date_range(sd, ed) 
    prices_all = get_data(syms, dates)  # automatically adds SPY 
    prices = prices_all[syms]  # only portfolio symbols 
    ms_trades = prices.copy()
    # Clear out the Dataframe so we can accumulate values into it
    for day in range(ms_trades.shape[0]):
        ms_trades.ix[day] = 0.0
    # Update with the manual strategy trades result
    ms_trades.update(ms_trades_temp)

    # Calculate the trades using Learner
    np.random.seed(1481090002)
    learner = sl.StrategyLearner(verbose = False, impact = impact) # constructor
    learner.addEvidence(symbol = symbol, sd=sd, ed=ed, sv = sv) # training phase
    sl_trades = learner.testPolicy(symbol = symbol, sd=sd, ed=ed, sv = sv) # testing phase

    # Construct trades for the benchmark
    # Benchmark: Investing in 1000 shares of "JPM" and holding that position.
    bm_trades = prices.copy()
    # Clear out the Dataframe so we can accumulate values into it
    for day in range(bm_trades.shape[0]):
        bm_trades.ix[day] = 0.0
    # Invest in 1000 shares on the 1st day
    bm_trades.ix[0] = 1000

    # Compute the Manual Strategy based Portfolio value
    ms_portvals = msim.compute_portvals(ms_trades, start_val = sv, commission=commission, impact=impact)
    # Compute the Strategy Learner based Portfolio value
    sl_portvals = msim.compute_portvals(sl_trades, start_val = sv, commission=commission, impact=impact)
    # Compute the Benchmark Portfolio value
    bm_portvals = msim.compute_portvals(bm_trades, start_val = sv, commission=commission, impact=impact)

    # Normalize the values
    normed_ms_portvals = ms_portvals / ms_portvals.values[0]
    normed_sl_portvals = sl_portvals / sl_portvals.values[0]
    normed_bm_portvals = bm_portvals / bm_portvals.values[0]

    # Plot graph for Manual Strategy Vs Strategy Learner
    normed_portvals = pd.concat([normed_sl_portvals,normed_ms_portvals, normed_bm_portvals], axis=1)
    normed_portvals.columns = ['Strategy Learner', 'Manual Strategy', 'Benchmark']
    ax_ms = normed_portvals.plot(title="Experiment 1", fontsize=12, linewidth=1, grid=True, color=['C1','C0','C2'])
    ax_ms.set_xlabel("Date")
    ax_ms.set_ylabel("Portfolio Value")
    plt.legend()
#    plt.savefig("experiment1.png")
#    plt.close()
    plt.show()

    # Compute Portfolio statistics for Manual Strategy
    ms_cum_ret, ms_avg_daily_ret, ms_std_daily_ret, ms_sharpe_ratio = msim.compute_stats(normed_ms_portvals)
    # Compute Portfolio statistics for Strategy Learner
    sl_cum_ret, sl_avg_daily_ret, sl_std_daily_ret, sl_sharpe_ratio = msim.compute_stats(normed_sl_portvals)
    # Compute Benchmark Statistics
    bm_cum_ret, bm_avg_daily_ret, bm_std_daily_ret, bm_sharpe_ratio = msim.compute_stats(normed_bm_portvals)
    # Compare portfolio against Benchmark 
    print "*********************************Experiment 1***********************************"
    print "***********Comparing Manual Strategy with Machine Learning Strategy*************"
    print('********************************In Sample Results*******************************')
    print "Date Range: {} to {}".format(sd, ed) 
    print 
    print "Strategy Learner - Sharpe Ratio of Portfolio: {}".format(sl_sharpe_ratio) 
    print "Manual Strategy - Sharpe Ratio of Portfolio: {}".format(ms_sharpe_ratio) 
    print "Benchmark - Sharpe Ratio of Portfolio: {}".format(bm_sharpe_ratio) 
    print 
    print "Strategy Learner - Cumulative Return of Portfolio : {}".format(sl_cum_ret) 
    print "Manual Strategy - Cumulative Return of Portfolio: {}".format(ms_cum_ret) 
    print "Benchmark - Cumulative Return of Portfolio : {}".format(bm_cum_ret) 
    print 
    print "Strategy Learner - Standard Deviation of Portfolio : {}".format(sl_std_daily_ret) 
    print "Manual Strategy - Standard Deviation of Portfolio: {}".format(ms_std_daily_ret) 
    print "Benchmark - Standard Deviation of Portfolio : {}".format(bm_std_daily_ret) 
    print 
    print "Strategy Learner - Average Daily Return of Portfolio : {}".format(sl_avg_daily_ret) 
    print "Manual Strategy - Average Daily Return of Portfolio: {}".format(ms_avg_daily_ret) 
    print "Benchmark - Average Daily Return of Portfolio : {}".format(bm_avg_daily_ret) 
    print 
    print "Strategy Learner - Final Portfolio Value: {}".format(sl_portvals[-1]) 
    print "Manual Strategy - Final Portfolio Value: {}".format(ms_portvals[-1]) 
    print "Benchmark - Final Portfolio Value: {}".format(bm_portvals[-1]) 