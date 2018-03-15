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

# use a support vector machine to identify peaks and valleys
# takes in a dataframe whose rows are days and cols are metrics:
# open, high, low, close, vwap, count, xrp volume, usd volume, buy volume
def identify_pv(df, alpha=0.055, delta=7, return_clf=True):

    X = df.values

    # scikit-learn's preprocessing feature to make data easier
    # for the classifier to group
    X = preprocessing.scale(X)

    # convert vwap to numpy array. Indices are days and elements are values
    vwap = df['vwap'].values
    y = []
    
    for index, price in enumerate(vwap):
        if index > delta - 1 and index < len(vwap) - (delta - 1):
            lin_reg = linear_model.LinearRegression()
            lin_X = np.arange(delta).reshape(-1, 1)
            
            lin_y = vwap[index - (delta - 1): index + 1]
            lin_reg.fit(lin_X, lin_y)
            past_coef = lin_reg.coef_

            lin_y = vwap[index: index + (delta)]
            lin_reg.fit(lin_X, lin_y)
            future_coef = lin_reg.coef_

            #print(past_coef, future_coef)
            
            # if the last 6 days have a positive trendline with a slope > a
            # and the next 6 days have a negative trendline with a slope < -a
            if past_coef < 0 and future_coef > 0 and (past_coef < -alpha or future_coef > alpha):
                y.append(-1)
            elif past_coef > 0 and future_coef < 0 and (past_coef > alpha or future_coef < -alpha):
                y.append(1)
            else:
                y.append(0)

    # trim x data be the same length as y data
    X = X[delta:-(delta - 1)]

    # postprocess classified data. If only 1 datapoint is recognized, the spike is too localized to
    # be indicative of a trend and should be discarded
    for index, val in enumerate(y):
        if index > 0 and index < len(y) - 1:
            if val != 0 and not (val == y[index - 1] or val == y[index + 1]):
                y[index] = 0

    if return_clf:
        clf = svm.SVC(probability=True)
        clf.fit(X, y)
        return clf
    else:
        return X, y

def predict(clf, X):
    predicted = clf.predict(X)
    return predicted

def plot(X, y, delta=7):
    plt.scatter(np.arange(len(y)), [abs(yut) for yut in y], color="green")

    plt.plot(df['vwap'][delta:-(delta - 1)], color="orange")
    plt.xticks(rotation='vertical')
    plt.subplots_adjust(bottom=0.05)
    plt.show()



df = get_data(150)

'''
TODO now: past features to x features (all daily features for past n days, and use all those to predict whether today will be pv)
'''

#X, y = identify_pv(df, return_clf=False)
#plot(X, y)




