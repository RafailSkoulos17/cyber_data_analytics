# coding: utf-8


import datetime
import os
import pickle
import time
from copy import deepcopy

import matplotlib.pyplot as plt
from sklearn import neighbors, svm
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
from collections import Counter
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import TomekLinks, EditedNearestNeighbours, OneSidedSelection
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.ensemble import BalancedRandomForestClassifier, RUSBoostClassifier
import sklearn.metrics as metrics
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import chi2


def plot_roc(fpr, tpr, roc_auc):
    import matplotlib.pyplot as plt
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


def plot_decision_tree(clf):
    from sklearn.tree import export_graphviz
    features = np.array(['txid', 'bookingdate', 'issuercountrycode', 'txvariantcode', 'bin', 'amount', 'currencycode',
                         'shoppercountrycode', 'shopperinteraction', 'simple_journal', 'cardverificationcodesupplied',
                         'cvcresponsecode', 'creationdate', 'accountcode', 'mail_id', 'ip_id', 'card_id',
                         'card_per_month', 'card_per_day',
                         'currency_per_month', 'merchant_per_month'])
    fearure_nums = [0, 4, 8, 10, 12, 13, 14, 15, 16]
    export_graphviz(clf, out_file='tree.dot', max_depth=5, feature_names=features[fearure_nums],
                    class_names=['non-fraudulent', 'fraudulent'],
                    filled=True, rounded=True,
                    special_characters=True, proportion=False,
                    precision=2)
    os.system('dot -Tpng tree.dot -o tree.png')


def make_clf(usx, usy, clf, clf_name, strategy='SMOTE', normalize=False):
    print('----------{}----------'.format(clf_name))
    totalTP, totalFP, totalFN, totalTN = 0, 0, 0, 0
    total_y_test = []
    total_y_prob = []
    j = 0
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    for train_index, test_index in skf.split(usx, usy):
        x_train, x_test = usx[train_index], usx[test_index]
        y_train, y_test = usy[train_index], usy[test_index]

        # select sampling strategy
        if strategy == 'SMOTE':
            sm = SMOTE(sampling_strategy=0.5, n_jobs=-1)
            x_train, y_train = sm.fit_resample(x_train, y_train)
            print('Resampled dataset shape %s' % Counter(y_train))
        elif strategy == 'ADASYN':
            ad = ADASYN(n_neighbors=50, n_jobs=-1)
            x_train, y_train = ad.fit_resample(x_train, y_train)
            print('Resampled dataset shape %s' % Counter(y_train))
        elif strategy == 'ENN':
            en = EditedNearestNeighbours(n_jobs=-1)
            x_train, y_train = en.fit_resample(x_train, y_train)
            print('Resampled dataset shape %s' % Counter(y_train))
        elif strategy == 'TL':
            tl = TomekLinks(sampling_strategy='auto', random_state=42, n_jobs=-1)
            x_train, y_train = tl.fit_resample(x_train, y_train)
            print('Resampled dataset shape %s' % Counter(y_train))
        elif strategy == 'OSS':
            oss = OneSidedSelection(n_jobs=-1)
            x_train, y_train = oss.fit_resample(x_train, y_train)
            print('Resampled dataset shape %s' % Counter(y_train))
        elif strategy == 'SMOTEK':
            sm = SMOTETomek(smote=SMOTE(sampling_strategy=0.5, n_jobs=-1), tomek=TomekLinks(n_jobs=-1))
            x_train, y_train = sm.fit_resample(x_train, y_train)
            print('Resampled dataset shape %s' % Counter(y_train))  # use this with threshold

        # normalize data if needed
        if normalize:
            scaler = StandardScaler().fit(x_train)
            x_train = scaler.transform(x_train)
            x_test = scaler.transform(x_test)

        clf.fit(x_train, y_train)

        # uncomment to plot the decision tree
        # if j == 0 and clf_name == 'DecisionTreeClassifier':
        #     plot_decision_tree(clf)
        # j += 1

        y_predict = clf.predict(x_test)
        y_predict_proba = clf.predict_proba(x_test)

        total_y_test += list(y_test)
        total_y_prob += list(y_predict_proba[:, 1])

        # modify the threshold for the two classes

        # for i in range(len(y_predict)):
        #     if y_predict[i] == 0 and y_predict_proba[i, 1] >= 0.4:
        #         y_predict[i] = 1

        # for i in range(len(y_predict)):
        #     if y_predict[i] == 1 and y_predict_proba[i, 1] <= 0.97:
        #         y_predict[i] = 0

        TP, FP, FN, TN = 0, 0, 0, 0
        for i in range(len(y_predict)):
            if y_test[i] == 1 and y_predict[i] == 1:
                TP += 1
            if y_test[i] == 0 and y_predict[i] == 1:
                FP += 1
            if y_test[i] == 1 and y_predict[i] == 0:
                FN += 1
            if y_test[i] == 0 and y_predict[i] == 0:
                TN += 1
        print('TP: ' + str(TP))
        print('FP: ' + str(FP))
        print('FN: ' + str(FN))
        print('TN: ' + str(TN))
        print(TP + TN + FP + FN)
        totalFN += FN
        totalFP += FP
        totalTN += TN
        totalTP += TP

    print('TOTAL TP: ' + str(totalTP))
    print('TOTAL FP: ' + str(totalFP))
    print('TOTAL FN: ' + str(totalFN))
    print('TOTAL TN: ' + str(totalTN))

    total_y_test = np.array(total_y_test)
    total_y_pred = np.array(total_y_prob)
    fpr, tpr, threshold = metrics.roc_curve(total_y_test, total_y_pred)
    roc_auc = metrics.auc(fpr, tpr)
    plot_roc(fpr, tpr, roc_auc, clf_name)


def aggregate_data(data):
    data['creationdate'] = pd.to_datetime(data['creationdate'], format='%Y-%m-%d %H:%M:%S')
    data['creation_month'] = data.creationdate.dt.month
    data['creation_year'] = data.creationdate.dt.year
    data['creation_day'] = data.creationdate.dt.day
    data['creation_hour'] = data.creationdate.dt.hour

    # number of transactions of specific card each creation day
    df = data[['card_id', 'creation_month']]
    df['card_month'] = 1
    card_per_month = df.groupby(['card_id', 'creation_month'], as_index=False)
    a = card_per_month.sum()
    data = pd.merge(data, a, how='inner')

    # number of transactions of specific card each creation day
    data['day'] = data.apply(lambda r: r['creationdate'].date(), axis=1)
    day = data[['card_id', 'day']]
    day['card_day'] = 1
    card_per_day = day.groupby(['card_id', 'day'], as_index=False)
    dd = card_per_day.sum()
    data = pd.merge(data, dd, how='inner')

    # currency type over month
    currency = data[['currencycode', 'creation_month']]
    currency['currency_month'] = 1
    cur_per_month = currency.groupby(['currencycode', 'creation_month'], as_index=False)
    cu = cur_per_month.sum()
    data = pd.merge(data, cu, how='inner')

    # Merchant type over month
    Merchant = data[['accountcode', 'creation_month']]
    Merchant['merchant_month'] = 1
    mer_per_month = Merchant.groupby(['accountcode', 'creation_month'], as_index=False)
    me = mer_per_month.sum()
    data = pd.merge(data, me, how='inner')

    return data


def data_encoding(data, column, threshold):
    count = dict(data[column].value_counts())
    mapping = {}
    for id in count.keys():
        if count[id] > threshold:
            mapping[id] = id
        else:
            mapping[id] = 'dc'
    data[column] = data[column].map(mapping)
    return data


if __name__ == "__main__":
    filename = 'data_for_student_case.csv'
    data = pd.read_csv(filename)

    # keep only valid samples
    data = data.loc[data['simple_journal'] != 'Refused']
    data = data.loc[data['bin'] != 'NA']
    data = data.loc[data['bin'] != 'na']
    data = data[~data['mail_id'].str.contains('emailNA')]
    # data['bookingdate'] = pd.to_datetime(data['bookingdate'], format='%Y-%m-%d %H:%M:%S')
    data['label'] = np.where(data['simple_journal'] == 'Chargeback', 1, 0)

    data = aggregate_data(deepcopy(data))

    conversion_dict = {'SEK': 0.09703, 'MXN': 0.04358, 'AUD': 0.63161, 'NZD': 0.58377, 'GBP': 1.13355}
    data['EuroAmount'] = data.apply(lambda r: int(round(conversion_dict[r['currencycode']] * r['amount'])), axis=1)

    # selected_features = ['issuercountrycode', 'txvariantcode', 'EuroAmount', 'amount',
    #                      'currencycode', 'shoppercountrycode', 'shopperinteraction', 'cardverificationcodesupplied',
    #                      'simple_journal',
    #                      'cvcresponsecode', 'accountcode', 'creation_hour', 'creation_day',
    #                      'creation_month', 'creation_year', 'ip_id',
    #                      'mail_id', 'bin', 'card_id', 'card_month', 'card_day', 'currency_month', 'merchant_month']

    selected_features = ['issuercountrycode', 'currencycode', 'shoppercountrycode', 'cvcresponsecode',
                         'accountcode', 'ip_id', 'card_month', 'card_day', 'currency_month', 'merchant_month', 'bin',
                         'card_id',
                         'label', 'simple_journal']

    # keep selected features
    data = data[selected_features]

    # data = data.dropna(axis=0, how='any')

    # convert bin to integer
    data['bin'] = data.apply(lambda r: int(round(r['bin'])), axis=1)

    # convert ip_id from categorical to number
    data['ip_id'] = data.apply(lambda r: int(float(r['ip_id'].replace('ip', ''))), axis=1)

    label = data['label']

    data = data.drop(
        ['label', 'simple_journal'], axis=1)

    # encode data
    data = data_encoding(deepcopy(data), 'bin', 3)
    data = data_encoding(deepcopy(data), 'card_id', 10)

    # one-hot encoding
    data = pd.get_dummies(data)
    
    x = data.values
    y = label.values

    # apply pca
    # print('Applying PCA')
    # pca = PCA(n_components=100)
    # x = pca.fit_transform(x)
    # print('PCA done')

    # apply feature selection with SelectKBest
    print('Applying SelectKBest')
    x = SelectKBest(chi2, k=100).fit_transform(x, y)
    print('SelectKBest done')

    # save features and labels
    with open('xy_all_2.pickle', 'wb') as handle:
        pickle.dump(x, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(y, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # load features and labels
    # with open('xy_all_2.pickle', 'rb') as handle:
    #     x = pickle.load(handle)
    #     y = pickle.load(handle)

    # clfs = {'RandomForestClassifier': RandomForestClassifier(n_estimators=10,  n_jobs=-1)}
    # clfs = {'LogisticRegression': LogisticRegression(solver='newton-cg',n_jobs=-1)}
    # clfs = {'RUSBoostClassifier': RUSBoostClassifier()}
    # clfs = {'AdaBoostClassifier': AdaBoostClassifier()}
    # clfs = {'DecisionTreeClassifier': DecisionTreeClassifier(criterion='entropy', class_weight={0: 1, 1: 10})}
    clfs = {'RandomForestClassifier': RandomForestClassifier(n_estimators=50, criterion='entropy',
                                                             class_weight={0: 1, 1: 10})}
    for clf_name, clf in clfs.items():
        usx = np.copy(x)
        usy = np.copy(y)
        if clf_name == 'LogisticRegression':
            make_clf(usx, usy, clf, clf_name, normalize=True, strategy='OSS')
        else:
            make_clf(usx, usy, clf, clf_name, strategy='OSS')
