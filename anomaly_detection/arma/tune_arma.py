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
from pca.utils import read_datasets

warnings.filterwarnings("ignore")


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
    return i - 1


scaled_df1, train_y1, scaled_df2, train_y2, scaled_test_df, y = read_datasets()

# coluns to drop, status signals
drop_columns = ['S_PU1', 'S_PU2', 'S_PU3', 'S_PU4', 'S_PU5', 'S_PU6', 'S_PU7', 'S_PU8', 'S_PU9', 'S_PU10', 'S_PU11',
                'S_V2'] + ['L_T1', 'L_T2', 'L_T3', 'L_T4', 'L_T5', 'L_T6', 'L_T7', 'F_PU1', 'F_PU2']

# drop columns on all the 3 datasets
train_data1 = scaled_df1.drop(drop_columns, axis=1)
train_data2 = scaled_df2.drop(drop_columns, axis=1)
test_data = scaled_test_df.drop(drop_columns, axis=1)

sensors = train_data1.columns.values

results = {}
for sensor in sensors:
    print(sensor)
    train_ts = pd.Series(scaled_df1[sensor], index=scaled_df1.index, )
    train_2_ts = pd.Series(scaled_df2[sensor], index=scaled_df2.index, )

    p = find_order(train_ts, terms='AR')
    q = find_order(train_ts, terms='MA')
    print('p={0}, q={1}'.format(p, q))
    if p == 0 or q == 0:
        p, q = 1, 1
    # TODO: maybe I should use train2 as tuning dataset
    try:
        res = sm.tsa.arma_order_select_ic(train_2_ts, ic=['aic'], max_ar=p, max_ma=q)
        order = res.aic_min_order
        order = list(order)
        if order[0] == 0 or order[1] == 0:
            order[0] = 1
            order[1] = 1
        order = tuple(order)
        results[sensor] = order
    except (np.linalg.LinAlgError, IndexError):
        if p == 0 or q == 0:
            p, q = 1, 1
        order = (p, q)
        results[sensor] = order
        print('{} has only 0s'.format(sensor))
    print("{0}, p={1} and q={2}".format(sensor, order[0], order[1]))
results = {k: [int(v[0]), int(v[1])] for k, v in results.items()}
with open("best_order.json", "w") as f:
    json.dump(results, f)
