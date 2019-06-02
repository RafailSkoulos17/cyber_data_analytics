import json
import pandas as pd
from statsmodels.tsa.stattools import pacf, acf
import statsmodels.api as sm
import numpy as np
import warnings
from utils import read_datasets

warnings.filterwarnings("ignore")


def find_order(data, terms):
    """
    Find the best p and q orders for ARMA based on acf and pacf values
    :param data: dataset
    :param terms: Specify if p or q need to be found
    :return: tuned order
    """
    # specify whether p or q need to be found
    if terms == "AR":
        lag = pacf(data, nlags=20, method='ols')
    elif terms == "MA":
        lag = acf(data, nlags=20)
    # define upper confidence interval as the one
    # of Normal distribution N(0, 1/len(data))
    thres = 1.96 / np.sqrt(len(data))
    for i, val in enumerate(lag):
        if val < thres:
            break
    return i - 1


def tune_arma():
    """
    Finds the best order for all sensors and save th ein a json file.
    """
    # read dataset
    scaled_df1, train_y1, scaled_df2, train_y2, scaled_test_df, y = read_datasets()

    # coluns to drop, status signals
    drop_columns = ['S_PU1', 'S_PU2', 'S_PU3', 'S_PU4', 'S_PU5', 'S_PU6', 'S_PU7', 'S_PU8', 'S_PU9', 'S_PU10', 'S_PU11',
                    'S_V2']

    # drop columns on all the 3 datasets
    train_data1 = scaled_df1.drop(drop_columns, axis=1)
    train_data2 = scaled_df2.drop(drop_columns, axis=1)

    # name of the sensors we use
    # sensors = train_data1.columns.values
    sensors = ['P_J280', 'F_PU3', 'F_V2', 'P_J300', 'P_J289', 'L_T6', 'F_PU10']

    results = {}
    for sensor in sensors:
        print(sensor)
        train_ts = pd.Series(train_data1[sensor], index=train_data1.index, )
        train_2_ts = pd.Series(train_data2[sensor], index=train_data2.index, )

        # find the orders from acf and pacf
        p = find_order(train_ts, terms='AR')
        q = find_order(train_ts, terms='MA')
        print('p={0}, q={1}'.format(p, q))
        if p == 0 or q == 0:
            p, q = 1, 1
        try:
            # find order by aic criterion, with thresholds for p and q defined by pacf and acf
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
        print("{0}, p={1} and q={2}".format(sensor, order[0], order[1]))
    results = {k: [int(v[0]), int(v[1])] for k, v in results.items()}
    with open("best_order.json", "w") as f:
        json.dump(results, f)


if __name__ == '__main__':
    tune_arma()
