import os
import time
from sklearn import neighbors
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from operator import itemgetter
from itertools import groupby
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import TomekLinks, EditedNearestNeighbours
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.ensemble import BalancedRandomForestClassifier, RUSBoostClassifier, EasyEnsembleClassifier, \
    BalancedBaggingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import export_graphviz


def string_to_timestamp(date_string):  # convert time string to float value
    time_stamp = time.strptime(date_string, '%Y-%m-%d %H:%M:%S')
    return time.mktime(time_stamp)


def plot_decision_tree(clf):
    features = np.array(['txid', 'bookingdate', 'issuercountrycode', 'txvariantcode', 'bin', 'amount', 'currencycode',
                         'shoppercountrycode', 'shopperinteraction', 'simple_journal', 'cardverificationcodesupplied',
                         'cvcresponsecode', 'creationdate', 'accountcode', 'mail_id', 'ip_id', 'card_id'])
    fearure_nums = [0, 4, 8, 10, 12]
    # dot_data = StringIO()
    export_graphviz(clf, out_file='tree.dot', max_depth=5, feature_names=features[fearure_nums],
                    class_names=['benign', 'fraudulent'],
                    filled=True, rounded=True,
                    special_characters=True, proportion=False,
                    precision=2)
    # graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    # Image(graph.create_png())
    os.system('dot -Tpng tree.dot -o tree.png')


def make_clf(usx, usy, clf, clf_name, normalize=False):
    print('----------{}----------'.format(clf_name))
    totalTP, totalFP, totalFN, totalTN = 0, 0, 0, 0
    total_y_test = []
    total_y_prob = []
    i = 0
    skf = StratifiedKFold(n_splits=10)
    for train_index, test_index in skf.split(usx, usy):
        x_train, x_test = usx[train_index], usx[test_index]
        y_train, y_test = usy[train_index], usy[test_index]

        # sm = SMOTE(k_neighbors=50, n_jobs=-1)
        # x_train, y_train = sm.fit_resample(x_train, y_train)
        # print('Resampled dataset shape %s' % Counter(y_train))

        # ad = ADASYN(n_neighbors=50, n_jobs=-1)
        # x_train, y_train = ad.fit_resample(x_train, y_train)
        # print('Resampled dataset shape %s' % Counter(y_train))

        # en = EditedNearestNeighbours(n_neighbors=3, n_jobs=-1)
        # x_train, y_train = en.fit_resample(x_train, y_train)
        # print('Resampled dataset shape %s' % Counter(y_train))

        tl = TomekLinks(sampling_strategy='auto', random_state=42, n_jobs=-1)
        x_train, y_train = tl.fit_resample(x_train, y_train)
        # print('Resampled dataset shape %s' % Counter(y_train))
        #
        # sm = SMOTETomek(smote=SMOTE(k_neighbors=50, n_jobs=-1), tomek=TomekLinks())
        # x_train, y_train = sm.fit_resample(x_train, y_train)
        # print('Resampled dataset shape %s' % Counter(y_train))

        # clf = BalancedRandomForestClassifier(n_estimators=100, n_jobs=-1)
        if normalize:
            scaler = StandardScaler().fit(x_train)
            x_train = scaler.transform(x_train)
            x_test = scaler.transform(x_test)

        clf.fit(x_train, y_train)

        if i == 0 and clf_name == 'DecisionTreeClassifier':
            plot_decision_tree(clf)

        y_predict = clf.predict(x_test)
        y_proba = clf.predict_proba(x_test)

        for i in range(len(y_predict)):
            if y_predict[i] and y_proba[i, 1] <= 0.65:
                y_predict[i] = 0

        for i in range(len(y_predict)):
            if y_test[i] == 1 and y_predict[i] == 1:
                totalTP += 1
            if y_test[i] == 0 and y_predict[i] == 1:
                totalFP += 1
            if y_test[i] == 1 and y_predict[i] == 0:
                totalFN += 1
            if y_test[i] == 0 and y_predict[i] == 0:
                totalTN += 1

    print('TOTAL TP: ' + str(totalTP))
    print('TOTAL FP: ' + str(totalFP))
    print('TOTAL FN: ' + str(totalFN))
    print('TOTAL TN: ' + str(totalTN))


if __name__ == "__main__":
    filename = 'original_data.csv'
    data = pd.read_csv(filename)
    x = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    # x_array = np.delete(x_array, [0, 1, 2, 3, 5, 6, 7, 9, 11, 13], 1)
    x = np.delete(x, [1, 2, 3, 5, 6, 7, 9, 11, 13], 1)

    # pca = PCA(n_components='mle', svd_solver='full')
    # pca.fit(x_array)
    # x_array = pca.transform(x_array)
    clfs = {'DecisionTreeClassifier': DecisionTreeClassifier(),
            'AdaBoostClassifier': AdaBoostClassifier(n_estimators=50)}
            # 'RandomForestClassifier': RandomForestClassifier(n_estimators=100)}
    for clf_name, clf in clfs.items():
        usx = np.copy(x)
        usy = np.copy(y)
        make_clf(usx, usy, clf, clf_name)
