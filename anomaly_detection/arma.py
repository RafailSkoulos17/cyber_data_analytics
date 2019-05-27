# TODO: modify threshold, add more sensors (p,q) even if all zero, maybe reduce p and q, LT2, PJ422, PJ14

import json

import pandas as pd
import scipy
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.stattools import adfuller, arma_order_select_ic
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np
import warnings
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")


def label_test_dataset():
    test_df = pd.read_csv('BATADAL_datasets/BATADAL_test_dataset.csv', index_col=0, parse_dates=[0],
                          date_parser=lambda x: pd.to_datetime(x, format="%d/%m/%y %H"))
    test_df.columns = test_df.columns.str.lstrip()

    test_df['label'] = 0
    test_df.loc['2017-01-16 09': '2017-01-19 06', 'label'] = 1
    test_df.loc['2017-01-30 08':'2017-02-02 00',  'label'] = 1
    test_df.loc['2017-02-09 03': '2017-02-10 09', 'label'] = 1
    test_df.loc['2017-02-12 01': '2017-02-13 07', 'label'] = 1
    test_df.loc['2017-02-24 05': '2017-02-28 08', 'label'] = 1
    test_df.loc['2017-03-10 14': '2017-03-13 21', 'label'] = 1
    test_df.loc['2017-03-25 20': '2017-03-27 01', 'label'] = 1
    y = test_df['label']  # separate the target values
    test_df.drop(['label'], axis=1, inplace=True)
    scaled_df = pd.DataFrame(StandardScaler().fit_transform(test_df.values), index=test_df.index,
                             columns=test_df.columns)

    return scaled_df, y


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


# def predict(coef, history):
#     yhat = 0.0
#     for i in range(1, len(coef) + 1):
#         yhat += coef[i - 1] * history[-i]
#     return yhat


def plot_residual(res, show_from=3500, show_to=4000):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(res)
    plt.show()


# def plot_anomalies(res):
#     a = [1 if i in anomalies else 0 for i in range(len(res))]
#     fig, ax = plt.subplots(figsize=(12, 8))
#     ax.plot(a)
#     plt.show()


def plot_anomalies(true_anomalies, predicted_anomalies, sensor):
    """
    Function that produces the lineplots for the visualization of the signals
    :param df: the dataframe with the signals
    :param features: which signals to plot
    :param dataset: the type of the dataset (training or test set)
    :param start_date: the starting date in the plot
    :param end_date: the ending date in the plot
    :return: creates and saves the produced lineplot
    """
    fig, ax = plt.subplots()
    ax.plot(true_anomalies.index, true_anomalies, label='True')
    ax.plot(true_anomalies.index, predicted_anomalies, label='Predicted')
    ax.xaxis.set_major_locator(mdates.DayLocator([5, 10, 15, 20, 25, 30]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
    ax.set_ylabel('Attacks')
    ax.grid()
    plt.legend()
    plt.title('Attacks for sensor {}'.format(sensor))
    plt.xticks(rotation=45)
    plt.xlabel("Date")
    plt.savefig('arma_{}.png'.format(sensor), bbox_inches='tight')


train_df1 = pd.read_csv('BATADAL_datasets/BATADAL_training_dataset1.csv', index_col=0, parse_dates=[0],
                        date_parser=lambda x: pd.to_datetime(x, format="%d/%m/%y %H"))
scaled_df1, train_y1 = scale_and_separate(train_df1)

train_df2 = pd.read_csv('BATADAL_datasets/BATADAL_training_dataset2.csv', index_col=0, parse_dates=[0],
                        date_parser=lambda x: pd.to_datetime(x, format="%d/%m/%y %H"))
train_df2.columns = train_df2.columns.str.lstrip()
scaled_df2, train_y2 = scale_and_separate(train_df2)

# test_df = pd.read_csv('BATADAL_datasets/BATADAL_test_dataset.csv', index_col=0, parse_dates=[0],
#                       date_parser=lambda x: pd.to_datetime(x, format="%d/%m/%y %H"))
# scaled_test_df, y = scale_and_separate(test_df, labels=False)
scaled_test_df, y = label_test_dataset()

with open("best_order.json", "r") as f:
    best_orders = json.load(f)
# sensors = ["F_PU6", "L_T6", "L_T2", "F_PU10", "F_PU1", "L_T1", "L_T4", "L_T5", "F_PU6", "L_T3", "L_T7"]
sensors = ["P_J422", "P_J14", "L_T5", "F_PU6", "L_T3", "L_T7"]

# for sensor in ['F_V2']:
for sensor in sensors:
    p = best_orders[sensor][0]
    q = best_orders[sensor][1]

    train_ts = pd.Series(scaled_df1[sensor], index=scaled_df1.index, )
    train_2_ts = pd.Series(scaled_df2[sensor], index=scaled_df2.index, )
    test_ts = pd.Series(scaled_test_df[sensor], index=scaled_test_df.index, )
    model = sm.tsa.SARIMAX(train_ts, order=(p, 0, q))  # 5,7 is decided using arma_order_select_ic method
    model_fit = model.fit(start_params = [0 for i in range(q+1)]+[1])
    # max_resid = np.max(np.abs(model_fit.resid))

    # test on training 2 to find the threshold value
    start = train_2_ts.index[0]
    predict = pd.Series()
    for t in train_2_ts.index:
        model2 = sm.tsa.SARIMAX(train_2_ts[start:t], order=(p, 0, q))
        model_fit2 = model2.filter(model_fit.params)
        predict = predict.append(model_fit2.forecast(1))

    resids2 = np.subtract(train_2_ts, predict)
    resids2 = abs(resids2 - np.mean(resids2))
    # min_resid = np.min(np.abs([x for i, x in enumerate(resids2) if train_y2[i] == 1]))
    r = [x for i, x in enumerate(resids2) if train_y2[i] == 1]
    if r:
        max_resid = np.max(np.abs(r))
    else:
        max_resid = np.max(np.abs(model_fit.resid))

    # test on test data
    start = test_ts.index[0]
    predict = pd.Series()
    for t in test_ts.index:
        model2 = sm.tsa.SARIMAX(test_ts[start:t], order=(p, 0, q))
        model_fit2 = model2.filter(model_fit.params)
        predict = predict.append(model_fit2.forecast(1))

    # max_resid = abs(np.max(predict-test_ts))
    include_from = 0
    res = np.subtract(test_ts[include_from:], predict[include_from:])
    res = abs(res - np.mean(res))

    # determine anomalies (threshold = 1.5*max_residual_on_training)
    predicted_anomalies = [1 if x > max_resid * 1.5 else 0 for i, x in enumerate(res)]
    # predicted_anomalies = [i + include_from if x > max_resid * 1.5 else 0 for i, x in enumerate(res)]
    TP = len([1 for x in y[predicted_anomalies] if x == 1])
    FP = len(predicted_anomalies) - TP
    print('Tp= {0}, FP= {1}'.format(TP, FP))
    # a = [1 if i in predicted_anomalies else 0 for i in range(len(res))]
    # plot_residual(res)
    # plot_anomalies(res)
    # plot_results()
    true_anomalies = y
    plot_anomalies(true_anomalies, predicted_anomalies, sensor)
