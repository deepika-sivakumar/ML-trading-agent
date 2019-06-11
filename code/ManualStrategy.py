# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 21:51:18 2019

@author: Deepika

"""
"""
ManualStrategy.py Code implementing a ManualStrategy object (your manual strategy). It should implement testPolicy() which returns a trades data frame (see below). The main part of this code should call marketsimcode as necessary to generate the plots used in the report.
"""
import datetime as dt 
from util import get_data, plot_data
import pandas as pd
import indicators as ind
import marketsimcode as msim
import numpy as np

class ManualStrategy(object):

    def __init__(self):
        pass # move along

    def author(self):
        return 'dsivakumar6'

    def testPolicy(self,symbol = "JPM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv = 100000):
        # Lookback window
        lookback = 14
        # Get the date range
        dates = pd.date_range(sd, ed)
        # Get the symbol in a list
        symbols = []
        symbols.append(symbol)
        symbol_SPY = []
        symbol_SPY.append("SPY")
    
         # Get all the prices of the symbols
        price = get_data(symbols, dates)  # automatically adds SPY
        # Forward fill and Backward fill
        price = price.fillna(method='ffill')
        price = price.fillna(method='bfill')
        price_symbol = price[symbols]  # only portfolio symbols 

        # Calculate the Technical Indicators (Price/SMA ratio, Bollinger Band Percentage, Stochastic Oscillator)
        sma = ind.calculate_price_SMA_ratio(price_symbol, symbols, lookback)
        bbp = ind.calculate_bbp(price_symbol, symbols, lookback)
        momentum = ind.calculate_momentum(price, symbols, lookback)

        # Create a binnary array showing when price is above SMA
        sma_cross = pd.DataFrame(0, index=sma.index, columns=sma.columns)
        sma_cross[sma >= 1] = 1
        # Turn that array into one that only shows the crossings (-1 is cross down, +1 is cross up)
        sma_cross[1:] = sma_cross.diff()
        sma_cross.ix[0] = 0

        # df_trades starts as a NAN array
        df_trades = price_symbol.copy()
        df_trades.ix[:,:] = np.NaN

        # Now calculate the df_trades
        # Stock may be oversold, BUY
        df_trades[(sma < 0.95) & (bbp < 0) & (momentum < -0.05) ] = 1000
        # Stock may be overbought, SELL
        df_trades[(sma > 1.05) & (bbp > 1) & (momentum > 0.05) ] = -1000
        # Apply exit order conditions
        df_trades[(sma_cross != 0)] = 0
        # All other days with NaN mean hold whatever you have, do nothing.

        # Forward fill NaNs with previous values, then fill remaining NaNs with 0
        df_trades.ffill(inplace=True)
        df_trades.fillna(0, inplace=True)
        # Now take the diff, which will give us an order to place only when the target shares changed.
        df_trades[1:] = df_trades.diff()
        df_trades.ix[0] = 0
        # Drop all rows with non-zero values (no orders)
        df_trades = df_trades.loc[(df_trades != 0).any(axis=1)]
        # Now we have only the days that have orders.
        return df_trades

    # Calculate the Benchmark using JPM
    def benchmark(self,sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv = 100000):
        # Get the date range
        dates = pd.date_range(sd, ed)
        # Get the symbol in a list
        symbols = []
        symbols.append("JPM")
         # Get all the prices of the symbols
        price = get_data(symbols, dates)  # automatically adds SPY
        price_JPM = price[symbols]  # only portfolio symbols 
        # Create the df_trades dataframe
        df_trades = price_JPM.copy()
        # Clear out the Dataframe so we can accumulate values into it
        for day in range(price.shape[0]):
            for sym in symbols:
                df_trades.ix[day,sym] = 0.0
        # Investing in 1000 shares of JPM and holding that position.
        df_trades.iloc[0] = +1000
        return df_trades

if __name__ == "__main__": 
    # Get the Object ManualStrategy
    ms = ManualStrategy()
    sd = dt.datetime(2008,1,1)
    ed = dt.datetime(2009,12,31)
    sv = 100000
    commission=9.95
    impact=0.005

    # Calculate trades using Manual Strategy
    df_trades = ms.testPolicy(symbol = "JPM", sd=sd, ed=ed, sv=sv) 
    # Calculate Long(BUY) & Short(SELL) entries seperately for plotting vertical lines
    long_trades = df_trades[df_trades > 0]
    long_trades = long_trades.dropna()
    short_trades = df_trades[df_trades < 0]
    short_trades = short_trades.dropna()

    # Calculate trades for the benchmark (holding 1000 JPM shares)
    df_trades_benchmark = ms.benchmark(sd=sd, ed=ed, sv=sv)

    # Compute the Rule-based Portfolio value
    df_portvals = msim.compute_portvals(df_trades, start_val = sv, commission=commission, impact=impact)
    # Compute the Benchmark Portfolio value
    df_portvals_benchmark = msim.compute_portvals(df_trades_benchmark, start_val = sv, commission=commission, impact=impact)
    # Normalize the values
    normed_portvals = df_portvals / df_portvals.values[0]
    normed_portvals_benchmark = df_portvals_benchmark / df_portvals_benchmark.values[0]

    # Plot graph for Manual Strategy
    ms_df = pd.concat([normed_portvals_benchmark, normed_portvals], axis=1)
    ms_df.columns = ['Benchmark','Portfolio']
    ax_ms = ms_df.plot(title="Manual Strategy - In Sample", fontsize=12, linewidth=1, grid=True, color=['C2','Red'])
    ax_ms.set_xlabel("Date")
    ax_ms.set_ylabel("Portfolio Value")
    ymin, ymax = ax_ms.get_ylim()
    ax_ms.vlines(x=long_trades.index,ymin=ymin, ymax=ymax,color='C0',label='Long')
    ax_ms.vlines(x=short_trades.index,ymin=ymin, ymax=ymax,color='black',label='Short')
    plt.legend()
#    plt.savefig("ms.png")
#    plt.close()
    plt.show()

    # Compute Portfolio statistics for Manual Strategy
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = msim.compute_stats(normed_portvals)
    # Compute Benchmark statistics
    cum_ret_benchmark, avg_daily_ret_benchmark, std_daily_ret_benchmark, sharpe_ratio_benchmark = msim.compute_stats(normed_portvals_benchmark)
    # Compare portfolio against Benchmark 
    print "******************Manual Strategy Results*******************"
    print('*************************In Sample Results*************************')
    print "Date Range: {} to {}".format(sd, ed) 
    print 
    print "Sharpe Ratio of Portfolio: {}".format(sharpe_ratio) 
    print "Sharpe Ratio of Benchmark : {}".format(sharpe_ratio_benchmark) 
    print 
    print "Cumulative Return of Portfolio: {}".format(cum_ret) 
    print "Cumulative Return of Benchmark : {}".format(cum_ret_benchmark) 
    print 
    print "Standard Deviation of Portfolio: {}".format(std_daily_ret) 
    print "Standard Deviation of Benchmark : {}".format(std_daily_ret_benchmark) 
    print 
    print "Average Daily Return of Portfolio: {}".format(avg_daily_ret) 
    print "Average Daily Return of Benchmark : {}".format(avg_daily_ret_benchmark) 
    print 
    print "Final Portfolio Value: {}".format(df_portvals[-1]) 
    print "Final Benchmark Value: {}".format(df_portvals_benchmark[-1]) 
    
    ################################### Comparitative analysis Out of Sample data ################################
    # The out of sample/testing period is January 1, 2010 to December 31 2011.
    sd_out = dt.datetime(2010,1,1)
    ed_out = dt.datetime(2011,12,31)

    # Calculate trades using Manual Strategy
    df_trades_out = ms.testPolicy(symbol = "JPM", sd=sd_out, ed=ed_out, sv=sv) 
    # Calculate Long(BUY) & Short(SELL) entries seperately for plotting vertical lines
    long_trades_out = df_trades_out[df_trades_out > 0]
    long_trades_out = long_trades_out.dropna()
    short_trades_out = df_trades_out[df_trades_out < 0]
    short_trades_out = short_trades_out.dropna()

    # Calculate trades for the benchmark (holding 1000 JPM shares)
    df_trades_benchmark_out = ms.benchmark(sd=sd_out, ed=ed_out, sv=sv)

    # Compute the Rule-based Portfolio value
    df_portvals_out = msim.compute_portvals(df_trades_out, start_val = sv, commission=commission, impact=impact)
    # Compute the Benchmark Portfolio value
    df_portvals_benchmark_out = msim.compute_portvals(df_trades_benchmark_out, start_val = sv, commission=commission, impact=impact)
    # Normalize the values
    normed_portvals_out = df_portvals_out / df_portvals_out.values[0]
    normed_portvals_benchmark_out = df_portvals_benchmark_out / df_portvals_benchmark_out.values[0]

    # Plot graph for Manual Strategy
    ms_df_out = pd.concat([normed_portvals_benchmark_out, normed_portvals_out], axis=1)
    ms_df_out.columns = ['Benchmark','Portfolio']
    ax_ms_out = ms_df_out.plot(title="Manual Strategy - Out of Sample", fontsize=12, linewidth=1, grid=True, color=['C2','Red'])
    ax_ms_out.set_xlabel("Date")
    ax_ms_out.set_ylabel("Portfolio Value")
    plt.legend()
#    plt.savefig("ms_out.png")
#    plt.close()
    plt.show()

    # Compute Portfolio statistics
    cum_ret_out, avg_daily_ret_out, std_daily_ret_out, sharpe_ratio_out = msim.compute_stats(normed_portvals_out)
    # Compute Benchmark statistics
    cum_ret_benchmark_out, avg_daily_ret_benchmark_out, std_daily_ret_benchmark_out, sharpe_ratio_benchmark_out = msim.compute_stats(normed_portvals_benchmark)
    # Compare portfolio against Benchmark JPM
    print('*************************Out of Sample Results*************************')
    print "Date Range: {} to {}".format(sd_out, ed_out) 
    print 
    print "Sharpe Ratio of Portfolio: {}".format(sharpe_ratio_out) 
    print "Sharpe Ratio of Benchmark : {}".format(sharpe_ratio_benchmark_out) 
    print 
    print "Cumulative Return of Portfolio: {}".format(cum_ret_out) 
    print "Cumulative Return of Benchmark : {}".format(cum_ret_benchmark_out) 
    print 
    print "Standard Deviation of Portfolio: {}".format(std_daily_ret_out) 
    print "Standard Deviation of Benchmark : {}".format(std_daily_ret_benchmark_out) 
    print 
    print "Mean of Daily Returns of Portfolio: {}".format(avg_daily_ret_out) 
    print "Mean of Daily Returns of Benchmark : {}".format(avg_daily_ret_benchmark_out) 
    print 
    print "Final Portfolio Value: {}".format(df_portvals_out[-1]) 
    print "Final Benchmark Value: {}".format(df_portvals_benchmark_out[-1]) 