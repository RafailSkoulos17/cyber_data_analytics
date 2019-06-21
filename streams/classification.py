import pandas as pd
import datetime
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from imblearn.over_sampling import SMOTE


def create_BClus(df):
    """
    Function that splits the netflow data into time windows and then aggregates them by source ip into aggregation
    windows in order for the BClus dataset to be constructed
    :param df: the initial dataframe
    :return: the BClus dataframe
    """
    t_start = df.index[0]
    new_data = pd.DataFrame()
    while t_start in df.index:
        # take a time window of 2 minutes
        t_end = t_start + datetime.timedelta(minutes=2)
        window = df.loc[(df.index >= t_start) & (df.index <= t_end)]

        # keep the remaining data
        remaining = df.loc[df.index > t_end]

        # loop for inner aggregation window
        agg_start = t_start
        for i in range(2):
            agg_end = agg_start + datetime.timedelta(minutes=1)
            agg_window = window.loc[(window.index >= agg_start) & (window.index <= agg_end)]

            # aggregate the data by source IP address
            src_groups = agg_window.groupby('src_ip')
            aggr = src_groups.aggregate({'packets': np.sum, 'bytes': np.sum, 'flows': np.sum})
            aggr['dst_ips'] = agg_window.groupby('src_ip').dst_ip.nunique()
            aggr['src_ports'] = agg_window.groupby('src_ip').src_port.nunique()
            aggr['dst_ports'] = agg_window.groupby('src_ip').dst_port.nunique()

            # and add them to the new dataset
            new_data = new_data.append(aggr, ignore_index=False)
            agg_start = agg_end

        if not len(remaining):
            break
        else:
            t_start = remaining.index[0]

    new_data = new_data.reset_index()
    return new_data


def check_infected(val, infected_ips):
    """
    Function to assign labels to instances given their ip
    :param val: the flow to be checked
    :param infected_ips: the list of infected hosts
    :return: 1 if val is in the infected list otherwise 0
    """
    return 1 if val in infected_ips else 0


def make_clf_cross(usx, usy, clf, clf_name, level):
    """
    Function for the classification task - Trains and tests the classifier clf using 10-fold cross-validation
    The sampling parameter sets the type of sampling to be used
    :param usx: the input instances
    :param usy: the labels of the instances
    :param clf: the classifier to be used
    :param clf_name: the name of the classifier (for plotting reasons)
    :param level: the evaluation level (for plotting reasons)
    :return: the classification results
    """
    print('---------- {} at {} level ----------'.format(clf_name, level))
    totalTP, totalFP, totalFN, totalTN = 0, 0, 0, 0
    j = 0
    skf = StratifiedKFold(n_splits=10, shuffle=True)  # apply 10-fold stratified cross validation
    for train_index, test_index in skf.split(usx, usy):

        # split data in training and test set
        x_train, x_test = usx[train_index], usx[test_index]
        y_train, y_test = usy[train_index], usy[test_index]

        # apply SMOTE for imbalance issues
        x_train, y_train = SMOTE(sampling_strategy=0.5).fit_resample(x_train, y_train)

        # fit the model and make predictions
        clf.fit(x_train, y_train)
        y_predict = clf.predict(x_test)

        for i in range(len(y_predict)):
            if y_test[i] and y_predict[i]:
                totalTP += 1
            if not y_test[i] and y_predict[i]:
                totalFP += 1
            if y_test[i] and not y_predict[i]:
                totalFN += 1
            if not y_test[i] and not y_predict[i]:
                totalTN += 1
        j += 1

    # just in case that no TP or FP are found
    if not (totalTP + totalFP):
        precision = 1
    else:
        precision = totalTP / (totalTP + totalFP)
    recall = totalTP / (totalTP + totalFN)
    accuracy = (totalTP + totalTN) / (totalTP + totalFN + totalTN + totalFP)
    print('TOTAL TP: ' + str(totalTP))
    print('TOTAL FP: ' + str(totalFP))
    print('TOTAL FN: ' + str(totalFN))
    print('TOTAL TN: ' + str(totalTN))
    print('TOTAL Accuracy: ' + str(accuracy))
    print('TOTAL Precision: ' + str(precision))
    print('TOTAL Recall: ' + str(recall))


if __name__ == '__main__':
    # if the data without the background are there, load them (again data from scenario 10 were used)
    data = pd.read_pickle('no_background_data.pkl')

    # resetting indices for data
    data = data.reset_index(drop=True)

    # parse packets and bytes as integers instead of strings
    data['packets'] = data['packets'].astype(int)
    data['bytes'] = data['bytes'].astype(int)

    # set date as index in the dataframe
    data = data.set_index(data.date)

    # Create BClus dataset
    bclus_data = create_BClus(data)

    # set the classifiers
    clf_name = 'RandomForestClassifier'
    clf = RandomForestClassifier(n_estimators=50, criterion='gini', class_weight='balanced')

    # name the infected hosts
    infected_ips = ['147.32.84.165', '147.32.84.191', '147.32.84.192', '147.32.84.193', '147.32.84.204',
                    '147.32.84.205', '147.32.84.206', '147.32.84.207', '147.32.84.208', '147.32.84.209']

    # enter the classification phase for each level
    eval_levels = ['packet', 'host']  # the 2 evaluation levels
    for level in eval_levels:
        # prepare the data according to the level
        final_data = bclus_data.copy()

        if level == 'host':
            final_data = final_data.groupby('src_ip').sum().reset_index()

        # label the processed dataset(s)
        final_data['label'] = final_data['src_ip'].apply(lambda z: check_infected(z, infected_ips))

        # separate the labels from the rest of the dataset
        y = final_data['label'].values
        x = final_data.drop(['src_ip', 'label'], axis=1).values

        # enter the classification phase
        print('Start the classification process')
        usx = np.copy(x)
        usy = np.copy(y)
        make_clf_cross(usx, usy, clf, clf_name, level)
