import itertools
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def read_datasets():
    """
    Reas the 3 datasets.
    :return: the 3 datasets and their labels
    """
    # read training 1 dataset
    train_df1 = pd.read_csv('../BATADAL_datasets/BATADAL_training_dataset1.csv', index_col=0, parse_dates=[0],
                            date_parser=lambda x: pd.to_datetime(x, format="%d/%m/%y %H"))
    scaled_df1, train_y1, scaler = scale_and_separate(train_df1)

    # read training 2 dataset
    train_df2 = pd.read_csv('../BATADAL_datasets/BATADAL_training_dataset2.csv', index_col=0, parse_dates=[0],
                            date_parser=lambda x: pd.to_datetime(x, format="%d/%m/%y %H"))
    train_df2.columns = train_df2.columns.str.lstrip()
    scaled_df2, _, _ = scale_and_separate(train_df2)
    scaled_df2, train_y2 = add_labels(scaled_df2, False)

    # read test dataset
    test_df = pd.read_csv('../BATADAL_datasets/BATADAL_test_dataset.csv', index_col=0, parse_dates=[0],
                          date_parser=lambda x: pd.to_datetime(x, format="%d/%m/%y %H"))
    scaled_test_df, _, _ = scale_and_separate(test_df, labels=False, test=True, scaler=scaler)

    scaled_test_df, y = add_labels(scaled_test_df, test=True)

    return scaled_df1, train_y1, scaled_df2, train_y2, scaled_test_df, y


def add_labels(df, test=True):
    """
    Adds labels to the training dataset 2
    :param df:
    :param test:
    :return: Scaled dataset and labels
    """
    df.columns = df.columns.str.lstrip()
    if test:
        df['ATT_FLAG'] = 0
        df.loc['2017-01-16 09': '2017-01-19 06', 'ATT_FLAG'] = 1
        df.loc['2017-01-30 08':'2017-02-02 00', 'ATT_FLAG'] = 1
        df.loc['2017-02-09 03': '2017-02-10 09', 'ATT_FLAG'] = 1
        df.loc['2017-02-12 01': '2017-02-13 07', 'ATT_FLAG'] = 1
        df.loc['2017-02-24 05': '2017-02-28 08', 'ATT_FLAG'] = 1
        df.loc['2017-03-10 14': '2017-03-13 21', 'ATT_FLAG'] = 1
        df.loc['2017-03-25 20': '2017-03-27 01', 'ATT_FLAG'] = 1
        y = df['ATT_FLAG']
        df.drop(['ATT_FLAG'], axis=1, inplace=True)
    else:
        df['ATT_FLAG'] = 0
        df.loc['2016-09-13 23': '2016-09-16 00', 'ATT_FLAG'] = 1
        df.loc['2016-09-26 11': '2016-09-27 10', 'ATT_FLAG'] = 1
        df.loc['2016-10-09 09': '2016-10-11 20', 'ATT_FLAG'] = 1
        df.loc['2016-10-29 19': '2016-11-02 16', 'ATT_FLAG'] = 1
        df.loc['2016-11-26 17': '2016-11-29 04', 'ATT_FLAG'] = 1
        df.loc['2016-12-06 07': '2016-12-10 04', 'ATT_FLAG'] = 1
        df.loc['2016-12-14 15': '2016-12-19 04', 'ATT_FLAG'] = 1
        y = df['ATT_FLAG']
        df.drop(['ATT_FLAG'], axis=1, inplace=True)
    return df, y


def scale_and_separate(df, labels=True, test=False, scaler=None):
    """
    Function that applies normalization to the data by subtracting the mean and dividing by the std for every
    observation in each column
    :param df: the initial dataframe
    :param labels: if there are attack labels in the dataset
    :param test: Boolean inicates if it's test dataset
    :param scaler: Scaler used for test dataset
    :return: the scaled dataset with the labels separated (if existed)
    """
    df.columns = df.columns.str.lstrip()
    y = None
    if labels:
        y = df['ATT_FLAG']
        df.drop(['ATT_FLAG'], axis=1, inplace=True)
    if test:
        scaled_df = pd.DataFrame(scaler.transform(df.values), index=df.index, columns=df.columns)
    else:
        scaler = StandardScaler().fit(df.values)
        scaled_df = pd.DataFrame(scaler.transform(df.values), index=df.index, columns=df.columns)
    return scaled_df, y, scaler


def get_blocks(y):
    """
    Finds the anomaly blocks
    :param y: Ground truth
    :return: Anomaly blocks
    """
    prev_val = -1
    blocks = []
    for ind, val in enumerate(y):
        if prev_val == 1 and val == 1:
            continue  # we're in the middle of a block
        elif prev_val != 1 and val == 1:
            start_block = ind  # we have found the beginning of the block
        elif prev_val == 1 and val != 1:
            end_block = ind  # we reached the end of the block
            blocks += [(start_block, end_block)]
        prev_val = val
    return blocks


def get_score(indices_positive_prediction, indices_true_positive, y):
    """
    Returns several scores based on the predictions done
    :param indices_positive_prediction: Indices of the hours that anomaly was predicted
    :param indices_true_positive: Indices of the hours that anomaly actually happened
    :param y: Ground truth
    :return: TP, FP, TN, FN, TPR, FNR, Sttd, Scm and S scores
    """

    blocks = get_blocks(y)

    indices_positive_prediction = sorted(indices_positive_prediction)

    detections = []
    # match the block detected, with the time it was detected, and the TTD/Dt value
    for ind in indices_positive_prediction:
        for block_ind, block in enumerate(blocks):
            if block[0] <= ind < block[1]:
                detections += [(block_ind, (ind - block[0]) / (block[1] - block[0]), ind)]
                break

    # group the detected attacks
    groups = [list(x) for (_, x) in itertools.groupby(detections, lambda x: x[0])]

    # choose the first time the attack was detected
    first_time_detected = [group[0] for group in groups]

    # compute the non detected attacks
    non_detected_attacks = (len(blocks) - len(groups))

    # compute Sttd, note that we add one for each non-detected attack
    # first_time_detected[1]  contains the TTD/Dt for each of the detected attacks
    Sttd = 1 - (sum(x[1] for x in first_time_detected) + non_detected_attacks) / len(blocks)

    # compute confusion matrix
    tp = len(set(indices_positive_prediction) & set(indices_true_positive))
    fp = len(set(indices_positive_prediction) - set(indices_true_positive))
    fn = len(set(indices_true_positive) - set(indices_positive_prediction))
    tn = len(y) - tp - fp - fn

    # compute TPR and TNR
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)

    # compute Scm
    Scm = (tpr + tnr) / 2
    gamma = 0.5  # default gamma used in competitions
    S = gamma * Sttd + (1 - gamma) * Scm  # compute final score
    return tp, fp, fn, tn, tpr, tnr, Sttd, Scm, S


def plot_anomalies(true_anomalies, predicted_anomalies):
    """
    Plots the predictes and true attacks
    :param true_anomalies: True attacks
    :param predicted_anomalies: Predicted attacks
    """
    fig, ax = plt.subplots()

    # convert predicted_anomalies to pandas Series for plotting reasons
    predicted_anomalies = pd.Series(predicted_anomalies, index=true_anomalies.index)

    true_anomalies.plot(label='True', kind='line', color='black', ax=ax)
    predicted_anomalies.plot(label='Predicted', kind='area', alpha=0.5, color='orange', ax=ax)
    ax.set_ylabel('Attacks')
    plt.legend()
    plt.title('Attacks')
    plt.xticks(rotation=45)
    plt.xlabel("Date")
    plt.savefig('../plots/pca/attacks_detected.png', bbox_inches='tight')
    # plt.show()