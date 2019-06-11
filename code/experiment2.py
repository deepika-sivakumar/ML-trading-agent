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
import matplotlib.pyplot as plt


def author():
    return 'dsivakumar6' # Georgia Tech username

if __name__=="__main__": 
    # Inputs
    symbol = "JPM"
    sd = dt.datetime(2008,1,1)
    ed = dt.datetime(2009,12,31)
    sv = 100000
    commission=0.00
    impact1 = 0.005
    impact2 = 0.015
    impact3 = 0.025

    np.random.seed(1481090002)
    # Calculate the trades using Learner with impact = 0.005
    learner1 = sl.StrategyLearner(verbose = False, impact = impact1) # constructor
    learner1.addEvidence(symbol = symbol, sd=sd, ed=ed, sv = sv) # training phase
    sl_trades1 = learner1.testPolicy(symbol = symbol, sd=sd, ed=ed, sv = sv) # testing phase

    # Compute the Strategy Learner based Portfolio value
    sl_portvals1 = msim.compute_portvals(sl_trades1, start_val = sv, commission=commission, impact=impact1)

    # Normalize the values
    normed_sl_portvals1 = sl_portvals1 / sl_portvals1.values[0]

    # Calculate the trades using Learner with impact = 0.015
    learner2 = sl.StrategyLearner(verbose = False, impact = impact2) # constructor
    learner2.addEvidence(symbol = symbol, sd=sd, ed=ed, sv = sv) # training phase
    sl_trades2 = learner2.testPolicy(symbol = symbol, sd=sd, ed=ed, sv = sv) # testing phase

    # Compute the Strategy Learner based Portfolio value
    sl_portvals2 = msim.compute_portvals(sl_trades2, start_val = sv, commission=commission, impact=impact2)

    # Normalize the values
    normed_sl_portvals2 = sl_portvals2 / sl_portvals2.values[0]

    # Calculate the trades using Learner with impact = 0.025
    learner3 = sl.StrategyLearner(verbose = False, impact = impact3) # constructor
    learner3.addEvidence(symbol = symbol, sd=sd, ed=ed, sv = sv) # training phase
    sl_trades3 = learner3.testPolicy(symbol = symbol, sd=sd, ed=ed, sv = sv) # testing phase

    # Compute the Strategy Learner based Portfolio value
    sl_portvals3 = msim.compute_portvals(sl_trades2, start_val = sv, commission=commission, impact=impact3)

    # Normalize the values
    normed_sl_portvals3 = sl_portvals3 / sl_portvals3.values[0]

    print "****************Experiment 2 - Effect of Impact Values on Learner Strategies*******************"
    # Plot graph for Effect of Impact on Metric 1 - Portfolio Values 
    normed_sl_portvals = pd.concat([normed_sl_portvals1, normed_sl_portvals2,normed_sl_portvals3], axis=1)
    normed_sl_portvals.columns = ['Impact = {}'.format(impact1), 'Impact = {}'.format(impact2), 'Impact = {}'.format(impact3)]
    ax_sl_portvals = normed_sl_portvals.plot(title="Experiment 2 - Effect of Impact on Portfolio Values", fontsize=12, linewidth=1, grid=True, color=['C2','C0','C1'])
    ax_sl_portvals.set_xlabel("Date")
    ax_sl_portvals.set_ylabel("Portfolio Value")
    plt.legend()
#    plt.savefig("experiment2_port.png")
#    plt.close()
    plt.show()

    # Calculate Long(BUY) & Short(SELL) entries seperately for plotting vertical lines
    long_trades1 = sl_trades1[sl_trades1 > 0]
    long_trades1 = long_trades1.dropna()
#    print('No of longtrades1 ::::',long_trades1.shape[0])
    short_trades1 = sl_trades1[sl_trades1 < 0]
    short_trades1 = short_trades1.dropna()

    long_trades2 = sl_trades2[sl_trades2 > 0]
    long_trades2 = long_trades2.dropna()
    short_trades2 = sl_trades2[sl_trades2 < 0]
    short_trades2 = short_trades2.dropna()
    
    long_trades3 = sl_trades3[sl_trades3 > 0]
    long_trades3 = long_trades3.dropna()
    short_trades3 = sl_trades3[sl_trades3 < 0]
    short_trades3 = short_trades3.dropna()

    no_long_trades = (long_trades1.shape[0], long_trades2.shape[0], long_trades3.shape[0])
    no_short_trades = (short_trades1.shape[0], short_trades2.shape[0], short_trades3.shape[0])
#    trades_chart(no_long_trades,no_short_trades)
    n_impacts = 3
    index = np.arange(n_impacts)
    bar_width = 0.15

    # Plot graph for Effect of Impact on Metric 2 - Trade Decisions
    fig_trades, ax_trades = plt.subplots()

    long_bars = ax_trades.bar(index, no_long_trades, bar_width,color='C0',label='Long Trades')
    short_bars = ax_trades.bar(index + bar_width, no_short_trades, bar_width,color='C1',label='Short Trades')

    ax_trades.set_title('Experiment 2 - Effect of Impact on Trade decisions')
    ax_trades.set_xlabel('Impact')
    ax_trades.set_ylabel('No of days traded')
    ax_trades.set_xticks(index + bar_width / 4)
    ax_trades.set_xticklabels(('Impact = {}'.format(impact1), 'Impact = {}'.format(impact2), 'Impact = {}'.format(impact3)))
    ax_trades.legend()
    #fig.tight_layout()
#    plt.savefig("experiment2_trades.png")
#    plt.close()
    plt.show()

    # Compute Portfolio statistics for Strategy Learner
    sl_cum_ret1, sl_avg_daily_ret1, sl_std_daily_ret1, sl_sharpe_ratio1 = msim.compute_stats(normed_sl_portvals1)
    sl_cum_ret2, sl_avg_daily_ret2, sl_std_daily_ret2, sl_sharpe_ratio2 = msim.compute_stats(normed_sl_portvals2)
    sl_cum_ret3, sl_avg_daily_ret3, sl_std_daily_ret3, sl_sharpe_ratio3 = msim.compute_stats(normed_sl_portvals3)

    # Compare portfolio for various impact values
    print('*************************In Sample Results*************************')
    print "Date Range: {} to {}".format(sd, ed) 
    print 
    print "Strategy Learner - Impact({})- Sharpe Ratio of Portfolio: {}".format(impact1,sl_sharpe_ratio1) 
    print "Strategy Learner - Impact({})- Sharpe Ratio of Portfolio: {}".format(impact2,sl_sharpe_ratio2) 
    print "Strategy Learner - Impact({})- Sharpe Ratio of Portfolio: {}".format(impact3,sl_sharpe_ratio3) 
    print 
    print "Strategy Learner - Impact({})- Cumulative Return of Portfolio : {}".format(impact1,sl_cum_ret1) 
    print "Strategy Learner - Impact({})- Cumulative Return of Portfolio : {}".format(impact2,sl_cum_ret2) 
    print "Strategy Learner - Impact({})- Cumulative Return of Portfolio : {}".format(impact3,sl_cum_ret3) 
    print 
    print "Strategy Learner - Impact({}) - Standard Deviation of Portfolio : {}".format(impact1,sl_std_daily_ret1) 
    print "Strategy Learner - Impact({}) - Standard Deviation of Portfolio : {}".format(impact2,sl_std_daily_ret2) 
    print "Strategy Learner - Impact({}) - Standard Deviation of Portfolio : {}".format(impact3,sl_std_daily_ret3) 
    print 
    print "Strategy Learner - Impact({}) - Average Daily Return of Portfolio : {}".format(impact1,sl_avg_daily_ret1) 
    print "Strategy Learner - Impact({}) - Average Daily Return of Portfolio : {}".format(impact2,sl_avg_daily_ret2) 
    print "Strategy Learner - Impact({}) - Average Daily Return of Portfolio : {}".format(impact3,sl_avg_daily_ret3) 
    print 
    print "Strategy Learner - Impact({}) - Final Portfolio Value: {}".format(impact1,sl_portvals1[-1]) 
    print "Strategy Learner - Impact({}) - Final Portfolio Value: {}".format(impact2,sl_portvals2[-1]) 
    print "Strategy Learner - Impact({}) - Final Portfolio Value: {}".format(impact3,sl_portvals3[-1]) 


