#!/usr/bin/env python
# coding: utf-8

# In[1]:


import ccxt
import time
import os
import datetime
import numpy as np
import pandas as pd

msec = 1000
minute = 60 * msec
hour = 60 * minute

symbol = 'BTC/USDT'
symbol_ = 'BTC_USDT'
market = 'binance'
timewindow = '5m'

if timewindow == '1h':
    offset = hour
    delay = offset / 1000
elif timewindow == '1m':
    offset = minute
    delay = offset / 1000
    
elif timewindow == '5m':
    offset = 5 * minute
    delay = offset / 1000


# In[2]:


def get_file_contents(filename):
    try:
        with open(filename, 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        print("'%s' file not found" % filename)


binance = ccxt.binance()
home_path = os.path.expanduser("~")
binance.apiKey = get_file_contents('./pub')
binance.secret = ""


# In[3]:


binance = ccxt.binance({
    'apiKey': binance.apiKey,
    'secret': binance.secret,
})
exchange = binance


# In[4]:


from_datetime = '2017-01-01 00:00:00'
from_timestamp = exchange.parse8601(from_datetime)

to_datetime = '2019-10-01 00:00:00'
# to_datetime = '2017-08-17 10:00:00'
to_timestamp = exchange.parse8601(to_datetime)


# In[5]:


t = exchange.fetch_trades(symbol=symbol, since=from_timestamp)


# In[6]:


# aggregate trades
#
#     [
#         {
#             "a": 26129,         # Aggregate tradeId
#             "p": "0.01633102",  # Price
#             "q": "4.70443515",  # Quantity
#             "f": 27781,         # First tradeId
#             "l": 27781,         # Last tradeId
#             "T": 1498793709153, # Timestamp
#             "m": True,          # Was the buyer the maker?
#             "M": True           # Was the trade the best price match?
#         }
#     ]
#


# In[7]:


buy_amount_list = []
sell_amount_list = []
for trade in t:
    if trade.get('side') == 'buy':
        buy_amount_list.append(trade.get('amount'))

    elif trade.get('side') == 'sell':
        sell_amount_list.append(trade.get('amount'))

buy_amount_np = np.array(buy_amount_list)
sell_amount_np = np.array(sell_amount_list)
N_buy = len(buy_amount_np)
N_sell = len(sell_amount_list)
buy_amount_avg = np.mean(buy_amount_np)
sell_amount_avg = np.mean(sell_amount_np)
buy_amount_std = np.std(buy_amount_np)
sell_amount_std = np.std(sell_amount_np)


# In[ ]:


df = pd.DataFrame()
header = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']

Total_buy_time_empty = 0
Total_sell_time_empty = 0
import time
start = time.time()


while from_timestamp < to_timestamp:
    try:
        ohlcvs = exchange.fetch_ohlcv(symbol, timewindow, from_timestamp, limit=1)
        
        # 1 group of trade is:
        t = exchange.fetch_trades(symbol=symbol, since=from_timestamp)
        buy_amount_list = []
        sell_amount_list = []
        price_list = []
        cost_list = []
        for trade in t:
            if trade.get('side') == 'buy':
                buy_amount_list.append(trade.get('amount'))

            elif trade.get('side') == 'sell':
                sell_amount_list.append(trade.get('amount'))
            
            price_list.append(trade.get('price'))
            cost_list.append(trade.get('cost'))
            

#         assert buy_amount_list != None, print('Not having any buy')
#         assert sell_amount_list != None, print('Not having any sell')
        if buy_amount_list == []:
            print('Not having any buyer')
            Total_buy_time_empty +=1
            buy_amount_list = [-1.0]
            
        if sell_amount_list == []:
            print('Not having any sell')
            Total_sell_time_empty +=1
            buy_amount_list = [-1.0]
            
        if price_list == []:
            price_list = [-1.0]
            
        if cost_list == []:
            cost_list = [-1.0]

        buy_amount_np = np.array(buy_amount_list)
        sell_amount_np = np.array(sell_amount_list)
        price_np = np.array(price_list)
        cost_np = np.array(cost_list)
            
        df_current = pd.DataFrame(ohlcvs, columns=header)
                
        df_current['N_buy'] = len(buy_amount_np)
        df_current['N_sell'] = len(sell_amount_list)
        df_current['buy_amount_avg'] = np.mean(buy_amount_np)
        df_current['sell_amount_avg'] = np.mean(sell_amount_np)
        df_current['buy_amount_std'] = np.std(buy_amount_np)
        df_current['sell_amount_std'] = np.std(sell_amount_np)
 
        df_current['price_avg'] = np.mean(price_np)
        df_current['price_std'] = np.std(price_np)
        
        df_current['cost_avg'] = np.mean(cost_np)
        df_current['cost_std'] = np.std(cost_np)
        df = df.append(df_current, ignore_index=True)

#         print(exchange.milliseconds(), 'Fetched', len(ohlcvs), 'candles')
        if len(ohlcvs) > 0:
            first = ohlcvs[0][0]
            last = ohlcvs[-1][0]
#             print('First candle epoch', first, exchange.iso8601(first))

            from_timestamp = ohlcvs[-1][0] + offset
            # v = ohlcvs[0][0]/ 1000
            # !date --date @{v} +"%Y-%m-%d %H:%M"
            print('Last candle epoch', last, exchange.iso8601(last))
        else:
            break

    except (ccxt.ExchangeError, ccxt.AuthenticationError, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout) as error:
        print('Got an error', type(error).__name__,
              error.args, ', retrying in', offset, 'seconds...')
        
done = time.time()
elapsed = done - start
print(elapsed, "secs")

print("empty buy  : ",Total_buy_time_empty)
print("empty sell : ",Total_sell_time_empty)
# In[ ]:


output_file = '{}/data/ccxt/{}_{}_{}.csv'.format(home_path,symbol_, market, timewindow)


# In[ ]:


print(df.shape)


# In[ ]:


df = df.dropna()
print(df.shape)


# In[ ]:


# df['MA_Close_240'] = df.rolling(240).mean().Close


# In[ ]:


output_file = './data/ccxt/extra/{}_{}_{}.csv'.format(symbol_, market, timewindow)
print(output_file)


# In[ ]:


df.to_csv(output_file)

