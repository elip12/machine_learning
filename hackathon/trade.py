import numpy as np
import pandas as pd
from sklearn import svm
from sklearn import preprocessing
import datetime as dt
import lin_reg

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

def add_lin_reg_prediction_to_df(df, duration=5):
    # convert timestamps to ints
    # run reg_model on last 5 ints and next int
    vwap_series = []
    xdata = [n for n in range(duration)]
    ydata = []
    for index, row in df.iterrows():
        if len(ydata) < duration:
            ydata.append(row['vwap'])
        else:
            m, b = lin_reg.best_fit(np.array(xdata), np.array(ydata))
            
            vwap_series.append(1 if m > 0 else 0)
            ydata.append(row['vwap'])
            ydata.pop(0)
    
    df = df[duration:]
    df = df.assign(linreg = pd.Series(vwap_series, index=df.index))
    
    return df


# get current metrics for xrp
# delta is the number of days before the present you want to get data from
# for now we are padding nan data, but should find a better thing to do later
def get_data(delta):
    
    now = dt.datetime.now()
    start = (now - dt.timedelta(days=delta)).strftime('%Y-%m-%dT%H:%M:%SZ')
    now = dt.datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')

    # this is the address that the aggregated price on xrpcharts.ripple.com uses
    # -- whose address is it? who fuckin knows
    address = 'https://data.ripple.com/v2/exchanges/XRP/USD+rvYAfWj5gh67oV6fW32ZzP3Aw4Eubs59B?limit=1000&format=csv&interval=1day&start=' + str(start) + '&end=' + str(now)
    df = pd.read_csv(address)
    df.drop(['base_issuer', 'counter_issuer', 'open_time', 'close_time', 'base_currency', 'counter_currency'], axis=1, inplace=True)
    df.rename(columns={'base_volume': 'xrp_volume', 'counter_volume': 'usd_volume'}, inplace=True)
    df.set_index('start', inplace=True)
    df.dropna(inplace=True)
    return df

# use scikit-learn's SVM to predict whether tomorrow's vwap will be higher than today's vwap
# extension: use derivatives not values to improve accuracy and recognize trends longer than 1 day
def classify(df, return_clf=True):

    X = df.values.tolist()
    X = preprocessing.scale(X)
    p_close = df['close'].tolist()
    y = []
    for index, close_price in enumerate(p_close):
        if index > 0 and close_price > p_close[index - 1]:
            y.append(1)
        elif index > 0 and close_price <= p_close[index - 1]:
            y.append(0)
    X = X[1:]

    if return_clf:
        clf = svm.SVC(probability=True)
        clf.fit(X, y)
        return clf
    else:
        return X, y

def predict(clf, X):
    predicted = clf.predict(X)
    return predicted


df = get_data(60)

test_df = df[45:]
classify_df = df[:45]
clf = classify(classify_df)
X, y = classify(test_df, return_clf=False)

without_lin = predict(clf, X).tolist()
cont = y

acc = 0
bcc = 0
for i in range(len(cont)):
    if without_lin[i] == cont[i]:
        acc += 1
    else:
        bcc += 1

print("acc: ", acc, "bcc: ", bcc, "t: ", acc / (acc + bcc))

# df = add_lin_reg_prediction_to_df(df, 3)

# test_df = df[45:]
# classify_df = df[:45]
# clf = classify(classify_df)
# X, y = classify(test_df, return_clf=False)

# with_lin = predict(clf, X).tolist()
# lin_cont = y

# acc = 0
# bcc = 0
# for i in range(len(lin_cont)):
#     if with_lin[i] == lin_cont[i]:
#         acc += 1
#     else:
#         bcc += 1

# print("\nlacc: ", acc, "lbcc: ", bcc, "t: ", acc / (acc + bcc))











