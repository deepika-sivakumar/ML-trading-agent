# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 21:51:10 2019

@author: Deepika

"""
"""
marketsimcode.py An improved version of your marketsim code that accepts a "trades" data frame (instead of a file). More info on the trades data frame below. It is OK not to submit this file if you have subsumed its functionality into one of your other required code files.
"""
import pandas as pd 
import numpy as np 
from util import get_data, plot_data

def author():
    return 'dsivakumar6'

"""
Function to convert the df_trades to Orders
"""
def convert_to_orders(df_trades):
    df_orders = df_trades.copy()
    # Create Columns "Symbol, Order, Shares"
    df_orders.columns = ['Symbol']
    df_orders['Order'] = 0
    df_orders['Shares'] = 0

    for index, row in df_trades.iterrows():
        orders_symbol = row.index[0]
        shares = row[row.index[0]]
        df_orders.ix[index,'Symbol'] = orders_symbol
        if(shares > 0):
            df_orders.ix[index,'Order'] = 'BUY'
            df_orders.ix[index,'Shares'] = shares
        elif(shares < 0):
            df_orders.ix[index,'Order'] = 'SELL'
            df_orders.ix[index,'Shares'] = shares * -1 # If its a SELL order, convert the negative to positive value for the df_orders
    return df_orders

"""
Function to Compute the Portfolio Value
"""
def compute_portvals(df_trades, start_val = 1000000, commission=9.95, impact=0.005): 
    # Convert the df_trades into Orders format
    df_orders = convert_to_orders(df_trades)

    # Sort the orders by date
    df_orders = df_orders.sort_index()

    # Get the start date & end date
    start_date = df_orders.index.min()
    end_date = df_orders.index.max() 
    dates = pd.date_range(start_date, end_date)

    # Get the symbols traded
    df_symbols_all = df_orders['Symbol']
    # Get the unique symbols alone
    df_symbols_temp = df_symbols_all.drop_duplicates()
    # Convert it into a list
    syms = df_symbols_temp.values.tolist()

    # Get all the prices of the symbols
    df_prices_all = get_data(syms, dates)  # automatically adds SPY
    df_prices = df_prices_all[syms]  # only portfolio symbols 

    # Add "Cash" column to the df_prices, initializing all values to 1.0
    df_prices['Cash'] = 1.0

    #Step2 - Trades Dataframe
    # Make a copy of the df_prices
    df_trades = df_prices.copy()
    df_trades[:] = 0

    # Iterate through the orders dataframe and calculate the shares & cash values for df_trades
    for index, row in df_orders.iterrows():
        orders_symbol = row['Symbol'] # Get the symbol
        order_type = row['Order'] # Get the Order type (BUY/SELL)
        symbol_price = df_prices.at[index, orders_symbol] # Get the Adjusted close price of that symbol on that date from df_prices
        # Set the No. of Shares traded from the df_orders to df_trades dataframe
        if(order_type == 'BUY'):
            # If its a "BUY" order we have gained the shares, so add
            df_trades.at[index, orders_symbol] = df_trades.at[index, orders_symbol] + row['Shares']
            #Calculate Cash = Sum_of_each_symbol_traded(No_Shares * Share_Price)
            trade_cost = row['Shares'] * symbol_price
            trade_impact = row['Shares'] * symbol_price * impact
            #If its a "BUY" order subtract (since we have spent that cash to buy)
            df_trades.at[index, 'Cash'] = df_trades.at[index, 'Cash'] - (trade_cost) - (trade_impact + commission)
        else:
            # If its a "SELL" order we have sold the shares, so subtract
            df_trades.at[index, orders_symbol] = df_trades.at[index, orders_symbol] - row['Shares']
            #Calculate Cash = Sum_of_each_symbol_traded(No_Shares * Share_Price)
            trade_cost = row['Shares'] * symbol_price
            trade_impact = row['Shares'] * symbol_price * impact
            #If its a "SELL" order add (since we have received that cash by selling)
            df_trades.at[index, 'Cash'] = df_trades.at[index, 'Cash'] + (trade_cost) - (trade_impact + commission)

    #Step 3 - Holdings Dataframe - How much asset value are you holding each day?
    df_holdings = df_trades.copy()
    # Add the Start Value to the First day cash
    df_holdings.at[start_date,'Cash'] = start_val + df_holdings.at[start_date,'Cash']
    # Add each row to the previous row values
    df_holdings = df_holdings.cumsum()

    #Step 4 df_values = df_prices * df_holdings
    df_values = df_prices * df_holdings

    #Step 5 Daily Portfolio Values
    df_portvals = df_values.sum(axis=1)

    return df_portvals

#Function to compute SPX port_vals
def compute_portvals_SPX(prices_SPX, start_value):
    normed = prices_SPX/prices_SPX.values[0]
    alloced = normed
    pos_value = alloced * start_value
    port_vals_SPX = pos_value.sum(axis=1)
    return port_vals_SPX

#Function to compute portfolio statistics
def compute_stats(port_vals):
#    daily_rets = compute_daily_returns(port_vals)
    daily_rets = (port_vals/port_vals.shift(1)) - 1
    cr = (port_vals[-1]/port_vals[0]) - 1
    adr = daily_rets.mean()
    sddr = daily_rets.std()
    sharpe_ratio=  adr/sddr
    sr = np.sqrt(252)*sharpe_ratio
    return cr, adr, sddr, sr

#Function to Compute the daily returns
def compute_daily_returns(port_vals):
    daily_returns = port_vals.copy()
    daily_returns[1:] =(port_vals[1:]/port_vals[:-1]) - 1
    daily_returns = daily_returns[1:]
    return daily_returns
