import json
import pandas as pd
import statsmodels.api as sm
import numpy as np
import warnings
from utils import read_datasets, get_score, plot_anomalies

warnings.filterwarnings("ignore")


def arma_detect():
    """
    Find the anomalies in 5 specific sensors with arma
    """
    # read datasets
    scaled_df1, train_y1, scaled_df2, train_y2, scaled_test_df, y = read_datasets()

    # initialize total anomalies (for all the selected sensors) to 0
    total_anomalies = pd.Series(
        [0 for _ in range(len(pd.Series(scaled_test_df["L_T2"], index=scaled_test_df.index, )))],
        index=scaled_test_df.index, )

    # coluns to drop, status signals
    drop_columns = ['S_PU1', 'S_PU2', 'S_PU3', 'S_PU4', 'S_PU5', 'S_PU6', 'S_PU7', 'S_PU8', 'S_PU9', 'S_PU10', 'S_PU11',
                    'S_V2']

    # drop columns on all the 3 datasets
    arma_train_data1 = scaled_df1.drop(drop_columns, axis=1)
    arma_train_data2 = scaled_df2.drop(drop_columns, axis=1)
    arma_test_data = scaled_test_df.drop(drop_columns, axis=1)

    # selected sensor names
    # sensors = arma_train_data1.columns.values
    sensors = ['P_J280', 'F_PU3', 'F_V2', 'P_J300', 'P_J289', 'L_T6', 'F_PU10']

    # get the best p and q as decided by tune_arma.py
    with open("best_order.json", "r") as f:
        best_orders = json.load(f)

    all_predictions_dict = {}

    # with open('arma_all.pickle', 'wb') as handle:
    #     pickle.dump(all_predictions_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    for sensor in sensors:
        # get p and q
        p = best_orders[sensor][0]
        q = best_orders[sensor][1]

        # get data in time series for the specific sensor
        train_ts = pd.Series(arma_train_data1[sensor], index=arma_train_data1.index, dtype=np.float)
        train_2_ts = pd.Series(arma_train_data2[sensor], index=arma_train_data2.index, dtype=np.float)
        test_ts = pd.Series(arma_test_data[sensor], index=arma_test_data.index, dtype=np.float)

        # fit arma on training dataset 1,
        # note that ARIMAX with order order=(p, 0, q) is equal to ARMA
        model = sm.tsa.SARIMAX(train_ts, order=(p, 0, q))  # 5,7 is decided using arma_order_select_ic method
        model_fit = model.fit(start_params=[0 for i in range(q + 1)] + [1], disp=False)

        # use training dataset 2 to find the threshold value
        start = train_2_ts.index[0]
        predict = pd.Series()
        # make predicition for training dataset 2
        # rrediction for an ARMA process with known coefficient is just a filtering problem
        for t in train_2_ts.index:
            model2 = sm.tsa.SARIMAX(train_2_ts[start:t], order=(p, 0, q))
            model_fit2 = model2.filter(model_fit.params)
            predict = predict.append(model_fit2.forecast(1))

        # find residuals
        resids2 = np.subtract(train_2_ts, predict)
        resids2 = np.abs(resids2)
        # resids2 = np.abs(resids2 - np.mean(resids2)) / np.std(resids2)

        # keep the max residual from the anomalies in the dataset
        anomaly_residuals = [x for i, x in enumerate(resids2) if train_y2[i] == 1]
        if anomaly_residuals:
            max_resid = np.max(np.abs(anomaly_residuals))
        else:
            max_resid = np.max(np.abs(model_fit.resid))

        # test on test data
        start = test_ts.index[0]
        predict = pd.Series()
        for t in test_ts.index:
            model2 = sm.tsa.SARIMAX(test_ts[start:t], order=(p, 0, q))
            model_fit2 = model2.filter(model_fit.params)
            predict = predict.append(model_fit2.forecast(1))
        # find residuals for test data
        res = np.subtract(test_ts, predict)
        res = np.abs(res)
        # res = np.abs(res - np.mean(res)) / np.std(res)

        # get predictions according to the threshold
        all_predictions = pd.Series([1 if x > max_resid else 0 for i, x in enumerate(res)], index=scaled_test_df.index)
        # find indices of anomalies
        predicted_anomalies = np.where(all_predictions > 0)[0]
        true_anomalies = np.where(y > 0)[0]
        # get score
        [tp, fp, fn, tn, tpr, tnr, Sttd, Scm, S] = get_score(predicted_anomalies, true_anomalies, y=y)
        print('---------- {} ----------'.format(sensor))
        print("TP: {0}, FP: {1}, TPR: {2}, TNR: {3}".format(tp, fp, tpr, tnr))
        print("Sttd: {0}, Scm: {1}, S: {2}".format(Sttd, Scm, S))
        # plot residual for the selected sensor
        plot_anomalies(y, all_predictions, sensor, 'arma')
        all_predictions_dict[sensor] = all_predictions
        # with open("arma_{}.pickle".format(sensor), "wb") as handle:
        #     pickle.dump(all_predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # combine predictions of sensors with logical or
        total_anomalies = total_anomalies | all_predictions

    # plot all anomalies
    plot_anomalies(y, total_anomalies, method='arma')
    # find indices of anomalies
    predicted_anomalies = np.where(total_anomalies == 1)[0]
    true_anomalies = np.where(y == 1)[0]
    # get score
    [tp, fp, fn, tn, tpr, tnr, Sttd, Scm, S] = get_score(predicted_anomalies, true_anomalies, y=y)
    print("TP: {0}, FP: {1}, TPR: {2}, TNR: {3}".format(tp, fp, tpr, tnr))
    print("Sttd: {0}, Scm: {1}, S: {2}".format(Sttd, Scm, S))


if __name__ == '__main__':
    arma_detect()
