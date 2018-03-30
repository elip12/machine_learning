import numpy as np
import pandas as pd
from sklearn import svm
from sklearn import preprocessing
from sklearn import datasets, linear_model
import datetime as dt
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import style
style.use('./elip12.mplstyle')
import matplotlib.pyplot as plt
import quandl

'''
metrics:
open: price at market open
high: high price of day
low: low price of day
close: price at close
vwap: volume weighted average price = (price/transaction * coin 
    traded/transaction) / total coin traded
volume: total coin traded per day
buy volume: total coin traded @ ask price (buyer 
    as opposed to bid/seller price)
'''
def get_btc(delta):
# so that I don't show my Quandl API key
    with open('/Users/Eli/Desktop/coding/quandl_api.txt', 'r') as api:
        key = api.read()
    quandl.ApiConfig.api_key = key

    now = dt.datetime.now()
    start = (now - dt.timedelta(days=delta)).strftime('%Y-%m-%d')

    # pull bitcoin price data from bitstamp through quandl
    #https://www.quandl.com/data/BCHARTS/BITSTAMPUSD-Bitcoin-Markets-bitstampUSD
    btc = quandl.get('BCHARTS/BITSTAMPUSD')
    btc = btc[:][start:]
    btc = np.round(btc, 2)
    btc.index = pd.to_datetime(btc.index)
    btc.index.rename('Date', inplace=True)

    btc = btc.rename(columns={"Weighted Price": 'vwap'})

    return btc

# p_open = df['open'].tolist()
# p_high = df['high'].tolist()
# p_low = df['low'].tolist()
# p_close = df['close'].tolist()
# p_vwap = df['vwap'].tolist()
# count = df['count'].tolist()
# v_xrp = df['xrp_volume'].tolist()
# v_usd = df['usd_volume'].tolist()
# v_buy = df['buy_volume'].tolist()

# get current metrics for xrp
# delta is the number of days before the present you want to get data from
# for now we are padding nan data, but should find a better thing to do later
def get_data(delta):
    
    now = dt.datetime.now()
    start = (now - dt.timedelta(days=delta)).strftime('%Y-%m-%dT%H:%M:%SZ')
    now = dt.datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')

    # this is the address that the aggregated price on xrpcharts.ripple.com uses
    address = 'https://data.ripple.com/v2/exchanges/XRP/USD+rvYAfWj5gh67oV6fW32ZzP3Aw4Eubs59B?limit=1000&format=csv&interval=1day&start=' + str(start) + '&end=' + str(now)
    df = pd.read_csv(address)
    df.drop(['base_issuer', 'counter_issuer', 'open_time', 'close_time', 'base_currency', 'counter_currency'], axis=1, inplace=True)
    df.rename(columns={'base_volume': 'xrp_volume', 'counter_volume': 'usd_volume'}, inplace=True)
    df.set_index('start', inplace=True)
    df.dropna(inplace=True)
    return df

# format dataframe
# each row of the dataframe now has 2 weeks worth of metrics 
def format_df(df, delta=14):

    X = df.values.tolist()
    
    for index in range(len(X) - 1, -1, -1):
        if index >= delta:
            for prev_day in range(1, delta):
                X[index].extend(X[index - prev_day])
            X[index] = np.asarray(X[index])

    # trim the head of X by delta elements
    X = np.asarray(X[delta:])
    
    return X

# use linear regression to identify peaks and valleys
# takes in a dataframe whose rows are days and cols are metrics:
# open, high, low, close, vwap, count, xrp volume, usd volume, buy volume
def identify_pv(df, alpha=0.055, delta=7):

    # convert vwap to numpy array. Indices are days and elements are values
    vwap = df['vwap'].values
    y = []
    
    for index, price in enumerate(vwap):
        if index >= delta and index <= len(vwap) - delta:
            lin_reg = linear_model.LinearRegression()
            lin_X = np.arange(delta).reshape(-1, 1)
            
            lin_y = vwap[index - (delta - 1): index + 1]
            lin_reg.fit(lin_X, lin_y)
            past_coef = lin_reg.coef_

            lin_y = vwap[index: index + (delta)]
            lin_reg.fit(lin_X, lin_y)
            future_coef = lin_reg.coef_

            #print(past_coef, future_coef)
            
            # if the last delta days have a positive trendline with a slope > alpha
            # or the next delta days have a negative trendline with a slope < -alpha
            if past_coef < 0 and future_coef > 0 and (past_coef < -alpha or future_coef > alpha):
                y.append(-1)
            elif past_coef > 0 and future_coef < 0 and (past_coef > alpha or future_coef < -alpha):
                y.append(1)
            else:
                y.append(0)


    # postprocess classified data. If only 1 datapoint is recognized, the spike is too localized to
    # be indicative of a trend and should be discarded
    for index, val in enumerate(y):
        if index > 0 and index < len(y) - 1:
            if val != 0 and not (val == y[index - 1] or val == y[index + 1]):
                y[index] = 0

    return y, delta
    

def classify(X, y):

    clf = svm.SVC()
    clf.fit(X, y)
    return clf

def predict(clf, X):
    predicted = clf.predict(X)
    return predicted

def plot(X, y, df, delta=7):
    plt.scatter(df.index[delta:-(delta - 1)], [abs(yut) * 10000 if yut >= 0 else 0 for yut in y], color="green")
    plt.scatter(df.index[delta:-(delta - 1)], [abs(yut) * 10000 if yut < 0 else 0 for yut in y], color="blue")

    plt.plot(df['vwap'][delta:-(delta - 1)], color="orange")
    plt.xticks(rotation='vertical')
    plt.subplots_adjust(bottom=0.05)
    plt.show()

X_delta = 14
y_delta = 7
n = 500

df = get_btc(600)
#print(df)

# df = get_data(20)
# print(df)
X = format_df(df, delta=X_delta)
y, delta = identify_pv(df, delta=y_delta)


X = X[:-y_delta + 1]
y = y[abs(X_delta - y_delta):]
y = np.asarray(y)

#print(y)
#print()
clf = classify(X[:n], y[:n])
pred = predict(clf, X[n:])

print(y[n:])
print()
print(pred)

# print(df)
#plot(X, y, df)




