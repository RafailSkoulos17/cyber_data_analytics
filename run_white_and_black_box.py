import os
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


def string_to_timestamp(date_string):  # convert time string to float value
    time_stamp = time.strptime(date_string, '%Y-%m-%d %H:%M:%S')
    return time.mktime(time_stamp)


def plot_decision_tree(clf):
    features = np.array(['issuercountry', 'txvariantcode', 'issuer_id', 'amount', 'currencycode',
                         'shoppercountry', 'interaction', 'verification', 'cvcresponse', 'creationdate_stamp',
                         'accountcode', 'mail_id', 'ip_id', 'card_id'])
    fearure_nums = [0, 4, 5, 8, 10, 12]
    # dot_data = StringIO()
    graph = Source(export_graphviz(clf, out_file=None, max_depth=5, feature_names=features[fearure_nums],
                    class_names=['benign', 'fraudulent'], filled=True, rounded=True, special_characters=True,
                    proportion=False, precision=2))
    png_bytes = graph.pipe(format='png')
    with open('dtree.png', 'wb') as f:
        f.write(png_bytes)
    # os.system('dot -Tpng tree.dot -o tree.png')


def make_clf(usx, usy, clf, clf_name, sampling, normalize=False):
    print('----------{}----------'.format(clf_name))
    totalTP, totalFP, totalFN, totalTN = 0, 0, 0, 0
    plot_ind = randint(0, 9)
    j = 0
    skf = StratifiedKFold(n_splits=10, shuffle=True)
    for train_index, test_index in skf.split(usx, usy):
        x_train, x_test = usx[train_index], usx[test_index]
        y_train, y_test = usy[train_index], usy[test_index]

        if sampling == 'SMOTE':
            x_train, y_train = SMOTE(sampling_strategy=0.6).fit_resample(x_train, y_train)
        elif sampling == 'ADASYN':
            x_train, y_train = ADASYN(sampling_strategy=0.6).fit_resample(x_train, y_train)
        elif sampling == 'ENN':
            x_train, y_train = EditedNearestNeighbours().fit_resample(x_train, y_train)
        elif sampling == 'Tomek':
            x_train, y_train = TomekLinks().fit_resample(x_train, y_train)
        elif sampling == 'SMOTETomek':
            x_train, y_train = SMOTETomek(sampling_strategy=0.6).fit_resample(x_train, y_train)
        elif sampling == 'SMOTEENN':
            x_train, y_train = SMOTEENN(sampling_strategy=0.5).fit_resample(x_train, y_train)
        elif sampling == 'NCR':
            x_train, y_train = NeighbourhoodCleaningRule().fit_resample(x_train, y_train)
        elif sampling == 'OSS':
            x_train, y_train = OneSidedSelection().fit_resample(x_train, y_train)

        if normalize:
            scaler = StandardScaler().fit(x_train)
            x_train = scaler.transform(x_train)
            x_test = scaler.transform(x_test)

        clf.fit(x_train, y_train)

        if plot_ind == j and clf_name == 'DecisionTreeClassifier':
            plot_decision_tree(clf)

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

    x = np.delete(x, [1, 2, 3, 6, 7, 9, 11, 13], 1)

    clfs = {'DecisionTreeClassifier': DecisionTreeClassifier(criterion='entropy', class_weight={0: 1, 1: 10})
            , 'RandomForestClassifier': RandomForestClassifier(n_estimators=50, criterion='entropy',
                                                               class_weight={0: 1, 1: 10})
            }
    for clf_name, clf in clfs.items():
        usx = np.copy(x)
        usy = np.copy(y)
        make_clf(usx, usy, clf, clf_name, 'OSS')
