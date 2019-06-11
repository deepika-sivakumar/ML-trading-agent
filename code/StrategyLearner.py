
import datetime as dt 
import pandas as pd 
import util as ut
import numpy as np
import random
import RTLearner as rt
import BagLearner as bl
import indicators as ind
from scipy import stats
import math

class StrategyLearner(object):

    def author(self):
        return 'dsivakumar6' # Georgia Tech username
 
    # constructor 
    def __init__(self, verbose = False, impact=0.0): 
        self.verbose = verbose 
        self.impact = impact 
        # Instantiate the learner here to remember the built tree
        self.learner = bl.BagLearner(learner = rt.RTLearner, kwargs = {"leaf_size":5, "verbose":False}, bags = 40, boost = False, verbose = False)
        # N days to calculate N-day returns
        self.ndays = 14
        # Lookback window for the indicators
        self.lookback = 14

    # Method to train the BagLearner containing an ensemble of RTLearners
    def addEvidence(self, symbol = "IBM", \
        sd=dt.datetime(2008,1,1), \
        ed=dt.datetime(2009,1,1), \
        sv = 10000): 

        # Get the past data to lookback and calculate the indicators
        sd_lookback = sd - dt.timedelta(days=self.lookback+15)
        ed_nday = ed + dt.timedelta(days=self.ndays+10)
        # Create training data
        syms=[symbol] 
        dates = pd.date_range(sd_lookback, ed_nday) 
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY 
        prices = prices_all[syms]  # only portfolio symbols 
        prices_SPY = prices_all['SPY']  # only SPY, for comparison later 

        # Forward fill and Backward fill
        prices = prices.fillna(method='ffill')
        prices = prices.fillna(method='bfill')

        # Calculate the Technical Indicators (Price/SMA ratio, Bollinger Band Percentage, Stochastic Oscillator)
        sma = ind.calculate_price_SMA_ratio(prices, syms, self.lookback)
        bbp = ind.calculate_bbp(prices, syms, self.lookback)
        momentum = ind.calculate_momentum(prices, syms, self.lookback)
        #Slice the indicators between the actual start date & end date
        sma = sma.loc[sd:ed]
        bbp = bbp.loc[sd:ed]
        momentum = momentum.loc[sd:ed]

        # Join all the indicators to form train X data
        trainX = np.concatenate((sma,bbp,momentum),axis=1)

        # Calculate N day returns as N day returns = Price[today+Ndays]/Price[today] - 1.0
        nday_ret = (prices.shift(-self.ndays)/prices) - 1.0
        # Slice the nday returns between the actual start & end dates
        nday_ret = nday_ret.loc[sd:ed]

        # Let us get the prices of the actual start date & end date
        prices_actual = prices.loc[sd:ed]
        YBUY = 0.02
        YSELL = -0.02
        # Construct train Y data
        trainY = prices_actual.copy()
        trainY = trainY.values * 0
        # Go LONG if returns > threshold
        trainY[nday_ret > YBUY+self.impact] = +1
        # Go SHORT if returns < threshold
        trainY[nday_ret < YSELL-self.impact] = -1
        # Convert trainY into a 1D numpy array
        trainY = np.hstack(trainY)

        # Train the BagLearner with data
        self.learner.addEvidence(trainX, trainY)

    # this method should use the existing policy and test it against new data 
    def testPolicy(self, symbol = "IBM", \
        sd=dt.datetime(2009,1,1), \
        ed=dt.datetime(2010,1,1), \
        sv = 10000): 

        # Get data for previous days to lookback and calculate the indicators
        sd_lookback = sd - dt.timedelta(days=self.lookback+15)
        # Get data for next days to calculate N day returns
        ed_nday = ed + dt.timedelta(days=self.ndays+10)

        # Get the prices
        syms = [symbol]
        dates = pd.date_range(sd_lookback, ed_nday) 
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY 
        prices = prices_all[syms]  # only portfolio symbols 

        # Forward fill and Backward fill
        prices = prices.fillna(method='ffill')
        prices = prices.fillna(method='bfill')

        # Calculate the Technical Indicators (Price/SMA ratio, Bollinger Band Percentage, Stochastic Oscillator)
        sma = ind.calculate_price_SMA_ratio(prices, syms, self.lookback)
        bbp = ind.calculate_bbp(prices, syms, self.lookback)
        momentum = ind.calculate_momentum(prices, syms, self.lookback)
        #Slice the indicators between the actual start date & end date
        sma = sma.loc[sd:ed]
        bbp = bbp.loc[sd:ed]
        momentum = momentum.loc[sd:ed]

        # Join all the indicators to form test X data
        testX = np.concatenate((sma,bbp,momentum),axis=1)

        # Query the Baglearner to get the trade predictions
        predY = self.learner.query(testX) # get the predictions

        # Let us get the prices of the actual start date & end date
        prices_actual = prices.loc[sd:ed]

        # Create the trades dataframe
        trades = prices_actual.copy()  # only portfolio symbols 
        trades_SPY = prices_all['SPY']  # only SPY, for comparison later 

        # Clear out the Dataframe so we can accumulate values into it
        for day in range(trades.shape[0]):
            trades.ix[day] = 0.0

        # Holdings (At any given day, the holdings should be only be either -1000, 0 or 1000)
        holdings = 0

        # Iterate through the dataframe to add trades
        for day in range(trades.shape[0]):
            # Go LONG if +1
            if (predY[day] > 0) and (holdings < 1000):
                # BUY 1000 shares first
                trades.ix[day] = 1000.0
                holdings = holdings + 1000
                # Even then, if the holdings sums upto only 0, BUY a 1000 more shares
                if (holdings == 0):
                    trades.ix[day] = trades.ix[day] + 1000
                    holdings = holdings + 1000
            # Go SHORT if -1
            elif (predY[day] < 0) and (holdings > -1000):
                # SELL 1000 shares first
                trades.ix[day] = -1000.0
                holdings = holdings - 1000
                # Even then, if the holdings comes to only 0, SELL a 1000 more shares
                if (holdings == 0):
                    trades.ix[day] = trades.ix[day] - 1000
                    holdings = holdings - 1000
        """
        trades.values[:,:] = 0 # set them all to nothing 
        trades.values[0,:] = 1000 # add a BUY at the start 
        trades.values[40,:] = -1000 # add a SELL 
        trades.values[41,:] = 1000 # add a BUY 
        trades.values[60,:] = -2000 # go short from long 
        trades.values[61,:] = 2000 # go long from short 
        trades.values[-1,:] = -1000 #exit on the last day 
        """
        if self.verbose: print type(trades) # it better be a DataFrame! 
        if self.verbose: print trades 
        if self.verbose: print prices_all 
        return trades 
 
if __name__=="__main__": 
    print "One does not simply think up a strategy" 
