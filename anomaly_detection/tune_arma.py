import json

import pandas as pd
import scipy
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.stattools import adfuller, arma_order_select_ic, pacf, acf
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np
import warnings


def scale_and_separate(df, labels=True):
    """
    Function that applies normalization to the data by subtracting the mean and dividing by the std for every
    observation in each column
    :param df: the initial dataframe
    :param labels: if there are attack labels in the dataset
    :return: the scaled dataset with the labels separated (if existed)
    """
    df.columns = df.columns.str.lstrip()
    y = None
    if labels:
        y = df['ATT_FLAG']
        df.drop(['ATT_FLAG'], axis=1, inplace=True)
    scaled_df = pd.DataFrame(StandardScaler().fit_transform(df.values), index=df.index, columns=df.columns)
    return scaled_df, y


def find_order(data, terms):
    if terms == "AR":
        lag = pacf(data, nlags=20, method='ols')
    elif terms == "MA":
        lag = acf(data, nlags=20)
    thres = 1.96 / np.sqrt(len(data))
    for i, val in enumerate(lag):
        if val < thres:
            break
    return (i - 1)


print('Reading the dataset')
train_df2 = pd.read_csv('BATADAL_datasets/BATADAL_training_dataset2.csv', index_col=0, parse_dates=[0],
                        date_parser=lambda x: pd.to_datetime(x, format="%d/%m/%y %H"))
train_df2.columns = train_df2.columns.str.lstrip()
print('Applying scaling on the data')
scaled_df2, train_y2 = scale_and_separate(train_df2)

test_df = pd.read_csv('BATADAL_datasets/BATADAL_test_dataset.csv', index_col=0, parse_dates=[0],
                      date_parser=lambda x: pd.to_datetime(x, format="%d/%m/%y %H"))
print('Applying scaling on the data')
scaled_test_df, _ = scale_and_separate(test_df, labels=False)

# sensors = ['L_T1', 'L_T2', 'L_T3', 'L_T4', 'L_T5', 'L_T6', 'L_T7', 'F_PU1', 'F_PU2', 'F_PU3',
#            'F_PU4', 'F_PU5', 'F_PU6', 'F_PU7', 'F_PU8', 'F_PU9', 'F_PU10', 'F_PU11', 'F_V2', 'P_J280', 'P_J269',
#            'P_J300', 'P_J256', 'P_J289', 'P_J415', 'P_J302', 'P_J306', 'P_J307', 'P_J317', 'P_J14',
#            'P_J422']
# sensors = ['F_PU1', 'L_T1', 'L_T4', 'L_T5', 'F_PU6', 'F_PU10']
# sensors = ['L_T6', 'F_PU6']
sensors = ['P_J422', 'P_J14']
results = {}
for sensor in sensors:
    try:
        print(sensor)
        train_ts = pd.Series(scaled_df2[sensor], index=scaled_df2.index, )

        p = find_order(train_ts, terms='AR')
        q = find_order(train_ts, terms='MA')
        print('p={0}, q={1}'.format(p, q))
        # TODO: maybe I should use train2 as tuning dataset
        res = sm.tsa.arma_order_select_ic(train_ts, ic=['aic'], max_ar=p, max_ma=q)
        order = res.aic_min_order
        results[sensor] = order

        # model2 = sm.tsa.SARIMAX(train_ts, order=(p, 0, q))
        # model_fit2 = model2.fit(disp=False)
        # resid = model_fit2.resid

        # fig = plt.figure(figsize=(12, 8))
        # ax1 = fig.add_subplot(211)
        # # you can use either data or ARMA with order=(0,0), its the same
        # fig = sm.graphics.tsa.plot_acf(resid.values.squeeze(), lags=40, ax=ax1)
        # fig = sm.graphics.tsa.plot_acf(train_ts.values.squeeze(), lags=40, ax=ax1)
        # ax2 = fig.add_subplot(212)
        # fig = sm.graphics.tsa.plot_pacf(resid, lags=40, ax=ax2)
        # fig = sm.graphics.tsa.plot_pacf(train_ts, lags=40, ax=ax2)
        # plt.show()
        # print('{} is OK'.format(sensor))
    except np.linalg.LinAlgError:
        print('{} has only 0s'.format(sensor))

results = {k: [int(v[0]), int(v[1])] for k, v in results.items()}
with open("best_order_2.json", "w") as f:
    json.dump(results, f)
