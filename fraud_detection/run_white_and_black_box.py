#!/usr/bin/python3

import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import TomekLinks, EditedNearestNeighbours, NeighbourhoodCleaningRule, OneSidedSelection
from imblearn.combine import SMOTETomek, SMOTEENN
from sklearn.preprocessing import StandardScaler
from sklearn.tree import export_graphviz
from graphviz import Source
from random import randint


def string_to_timestamp(date_string):
    '''
    Function coverting a time string to a float timestamp
    '''
    time_stamp = time.strptime(date_string, '%Y-%m-%d %H:%M:%S')
    return time.mktime(time_stamp)


def plot_decision_tree(clf):
    '''
    Function for the classification task - Plots the structure of the trained decision tree
    '''
    features = np.array(['issuercountry', 'txvariantcode', 'issuer_id', 'amount', 'currencycode',
                         'shoppercountry', 'interaction', 'verification', 'cvcresponse', 'creationdate_stamp',
                         'accountcode', 'mail_id', 'ip_id', 'card_id'])
    fearure_nums = [0, 4, 5, 8, 10, 12]
    graph = Source(export_graphviz(clf, out_file=None, max_depth=3, feature_names=features[fearure_nums],
                    class_names=['benign', 'fraudulent'], filled=True, rounded=True, special_characters=True,
                    proportion=False, precision=2))
    png_bytes = graph.pipe(format='png')
    with open('dtree.png', 'wb') as f:
        f.write(png_bytes)


def make_clf(usx, usy, clf, clf_name, sampling, normalize=False):
    '''
    Function for the classification task - Trains and tests the classifier clf using 10-fold cross-validation
    If normalize flag is True then the data are being normalised
    The sampling parameter sets the type of sampling to be used
    '''
    print('----------{} with {}----------'.format(clf_name, sampling))
    totalTP, totalFP, totalFN, totalTN = 0, 0, 0, 0
    plot_ind = randint(0, 9)
    j = 0
    skf = StratifiedKFold(n_splits=10, shuffle=True)
    for train_index, test_index in skf.split(usx, usy):
        x_train, x_test = usx[train_index], usx[test_index]
        y_train, y_test = usy[train_index], usy[test_index]

        if sampling == 'SMOTE':
            x_train, y_train = SMOTE(sampling_strategy=0.3).fit_resample(x_train, y_train)
        elif sampling == 'ADASYN':
            x_train, y_train = ADASYN(sampling_strategy=0.3).fit_resample(x_train, y_train)
        elif sampling == 'ENN':
            x_train, y_train = EditedNearestNeighbours().fit_resample(x_train, y_train)
        elif sampling == 'Tomek':
            x_train, y_train = TomekLinks().fit_resample(x_train, y_train)
        elif sampling == 'SMOTETomek':
            x_train, y_train = SMOTETomek(sampling_strategy=0.3).fit_resample(x_train, y_train)
        elif sampling == 'SMOTEENN':
            x_train, y_train = SMOTEENN(sampling_strategy=0.3).fit_resample(x_train, y_train)
        elif sampling == 'NCR':
            x_train, y_train = NeighbourhoodCleaningRule().fit_resample(x_train, y_train)
        elif sampling == 'OSS':
            x_train, y_train = OneSidedSelection().fit_resample(x_train, y_train)

        if normalize:
            scaler = StandardScaler().fit(x_train)
            x_train = scaler.transform(x_train)
            x_test = scaler.transform(x_test)

        clf.fit(x_train, y_train)

        # if plot_ind == j and clf_name == 'DecisionTreeClassifier':
        #     plot_decision_tree(clf)

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

    print('TOTAL TP: ' + str(totalTP))
    print('TOTAL FP: ' + str(totalFP))
    print('TOTAL FN: ' + str(totalFN))
    print('TOTAL TN: ' + str(totalTN))


if __name__ == "__main__":
    filename = 'original_data.csv'
    data = pd.read_csv(filename)
    x = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    x = np.delete(x, [1, 2, 3, 6, 7, 9, 11, 13], 1)  # specific features are kept

    # dictionaries with the two classifiers to be tested
    clfs = {'DecisionTreeClassifier': DecisionTreeClassifier(criterion='entropy', class_weight='balanced')
            , 'RandomForestClassifier': RandomForestClassifier(n_estimators=50, criterion='entropy',
                                                               class_weight='balanced')
            }
    for smlp in ['SMOTE', 'ADASYN', 'Tomek', 'OSS', 'ENN', 'SMOTETomek', 'SMOTEENN']:  # check different types of sampling
        for clf_name, clf in clfs.items():
            usx = np.copy(x)
            usy = np.copy(y)
            make_clf(usx, usy, clf, clf_name, smlp)
