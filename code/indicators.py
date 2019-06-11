# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 21:50:15 2019

@author: Deepika

"""
"""
indicators.py Your code that implements your indicators as functions that operate on dataframes. 
The "main" code in indicators.py should generate the charts that illustrate your indicators in the report.
"""
import datetime as dt 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from util import get_data, plot_data

import warnings
warnings.filterwarnings("ignore")

def author():
    return 'dsivakumar6'

"""
Function to Calculate the Simple Moving Average(SMA)
"""
def calculate_SMA(price, symbols, lookback):
    # Calculate SMA for the entire date range for the symbols
    sma = price.copy()
    # Clear out the Dataframe so we can accumulate values into it
    for day in range(price.shape[0]):
        for sym in symbols:
            sma.ix[day,sym] = 0
    # Calculate SMA as the rolling mean using pandas
    sma = price.rolling(lookback).mean()
    # Fill nan for days before lookback
    for day in range(price.shape[0]):
        # This day is too early to calculate the full SMA
        if day < lookback:
            for sym in symbols:
                sma.ix[day,sym] = np.nan
            continue
    return sma

"""
Function to Calculate Price/SMA ratio
"""
def calculate_price_SMA_ratio(price, symbols, lookback):
    # Calculate the SMA first
    sma = calculate_SMA(price, symbols, lookback)

    # Turn SMA into Price/SMA ratio
    for day in range(lookback, price.shape[0]):
        for sym in symbols:
            sma.ix[day,sym] = price.ix[day,sym] / sma.ix[day,sym]
    return sma

"""
Function to Calculate the Rolling Standard Deviation
"""
def calculate_RSTD(price, symbols, lookback):
    # Calculate Rolling Std Dev for the entire date range for the symbols
    rstd = price.copy()
    # Clear out the Dataframe so we can accumulate values into it
    for day in range(price.shape[0]):
        for sym in symbols:
            rstd.ix[day,sym] = 0
    # Calculate RSTD as the rolling standard deviation using pandas
    rstd = price.rolling(lookback).std()
    # Fill nan for days before lookback
    for day in range(price.shape[0]):
        # This day is too early to calculate the full SMA
        if day < lookback:
            for sym in symbols:
                rstd.ix[day,sym] = np.nan
            continue
    return rstd

"""
Function to calculate upper Bollinger Band
"""
def calculate_upper_bb(sma, rstd):
    upper_band = sma + (2 * rstd)
    return upper_band

"""
Function to calculate upper Bollinger Band
"""
def calculate_lower_bb(sma, rstd):
    lower_band = sma - (2 * rstd)
    return lower_band

"""
Function to Calculate Bollinger Band Percentage(BBP)
"""
def calculate_bbp(price, symbols, lookback):
    # Calculate the SMA first
    sma = calculate_SMA(price,symbols,lookback)

    # Calculate the Rolling Standard Deviation
    rstd = calculate_RSTD(price, symbols, lookback)

    # Calculate the Upper & Lower Bollinger Bands
    upper_band = calculate_upper_bb(sma, rstd)
    lower_band = calculate_lower_bb(sma, rstd)

    # Calculate BB for the entire date range for the symbols
    bbp = price.copy()
    # Clear out the Dataframe so we can accumulate values into it
    for day in range(price.shape[0]):
        for sym in symbols:
            bbp.ix[day,sym] = 0

    # Calculate Bollinger Band Percentage (BBP)
    bbp = (price - lower_band) / (upper_band - lower_band)
    return bbp

"""
Function to Calculate the Momentum
"""
def calculate_momentum(price, symbols, lookback):
    momentum = (price / price.shift(lookback)) - 1
    return momentum

"""
Function to Calculate the Stochastic Oscillator
"""
def calculate_stochastic_oscillator(price, symbols, lookback):
    # Calculate Stochastic Oscillator values for the entire date range for the symbols
    stoch = price.copy()
    # Clear out the Dataframe so we can accumulate values into it
    for day in range(price.shape[0]):
        for sym in symbols:
            stoch.ix[day,sym] = 0
    highest_high = price.rolling(lookback).max()
    lowest_low = price.rolling(lookback).min()

    # %K = (C - L20)/(H20 - L20) * 100
    stoch = ((price - lowest_low) / (highest_high - lowest_low)) * 100

    # Fill nan for days before lookback
    for day in range(price.shape[0]):
        # This day is too early to calculate the full SMA
        if day < lookback:
            for sym in symbols:
                stoch.ix[day,sym] = np.nan
            continue
    return stoch

def test_code():
    # Get the start date & end date
    start_date = dt.datetime(2008,1,1)
    end_date = dt.datetime(2009,12,31)

    dates = pd.date_range(start_date, end_date)

    # Symbols to trade
    symbols = ['JPM']

    # Get all the prices of the symbols
    price_all = get_data(symbols, dates)  # automatically adds SPY
    price = price_all[symbols]  # only portfolio symbols 

    # Lookback window is 14 days
    lookback = 14

    # Calculate the Normalized Price
    normed_price = price/price.values[0]

    # Calculate the Simple Moving Average
    sma = calculate_SMA(normed_price, symbols, lookback)
    # Calculate the Rolling Standard Deviation
    rstd = calculate_RSTD(normed_price, symbols, lookback)
    price_SMA_ratio = calculate_price_SMA_ratio(normed_price, symbols, lookback)

    # Calculate the Upper & Lower Bollinger Bands
    upper_band = calculate_upper_bb(sma, rstd)
    lower_band = calculate_lower_bb(sma, rstd)
    # Calculate Bollinger Band Percentage
    bbp = calculate_bbp(normed_price, symbols, lookback)

    # Calculate the Momentum
    momentum = calculate_momentum(normed_price, symbols, lookback)

    # Calculate the Stochastic Oscillator
#    stoch = calculate_stochastic_oscillator(normed_price, symbols, lookback)

#    print('*************************Price*************************')
#    print(price)
#    print('************************normed_price*******************')
#    print(normed_price)
#    print('************************sma***********************')
#    print(sma)
#    print('**************************Price/SMA Ratio********************')
#    print(price_SMA_ratio)
#    print('**************************Bollinger Band %********************')
#    print(bbp)
#    print('**************************Stochastic Oscillator********************')
#    print(stoch)
#    print('*************************Momentum*************************')
#    print(momentum)

    # Plot the Indicators
    # Plotting Price/SMA ratio indicator
    price_SMA_df = pd.concat([normed_price, sma, price_SMA_ratio], axis=1)
    price_SMA_df.columns = ['Normalized Price', 'SMA', 'Price/SMA']
    ax_price_SMA = price_SMA_df.plot(title="Technical Indicator: Price/SMA Ratio", fontsize=12, linewidth=1, grid=True)
    ax_price_SMA.set_xlabel("Date")
#    plt.savefig("price_sma_ratio.png")
#    plt.close()
    plt.show()

    # Plotting Bollinger Band Percentage Indicator
    bbp_df = pd.concat([normed_price, sma, upper_band, lower_band, bbp], axis=1)
    bbp_df.columns = ['Normalized Price', 'SMA', 'Upper Band', 'Lower Band', 'BBP']
    ax_bbp = bbp_df.plot(title="Technical Indicator: Bollinger Band Percentage(BBP)", fontsize=12, linewidth=1, grid=True,color=['C0','C1', 'Red', 'C2', 'gold'])
    ax_bbp.set_xlabel("Date")
#    plt.savefig("bbp.png")
#    plt.close()
    plt.show()

    # Plotting Momentum indicator
    momentum_df = pd.concat([normed_price, momentum], axis=1)
    momentum_df.columns = ['Normalized Price', 'Momentum']
    ax_momentum = momentum_df.plot(title="Technical Indicator: Momentum", fontsize=12, linewidth=1, grid=True)
    ax_momentum.set_xlabel("Date")
#    plt.savefig("momentum.png")
#    plt.close()
    plt.show()

#    # Plotting Stochastic Oscillator indicator
#    momentum_df = pd.concat([normed_price, stoch], axis=1)
#    momentum_df.columns = ['Price', 'Stochastics']
#    ax_momentum = momentum_df.plot(title="Technical Indicator: Stochastic Oscillator", fontsize=12, linewidth=1, grid=True)
#    ax_momentum.set_xlabel("Date")
#    plt.show()

if __name__ == "__main__": 
    test_code() 
