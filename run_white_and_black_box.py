# coding: utf-8

# In[7]:


import datetime
import os
import time
import matplotlib.pyplot as plt
from sklearn import neighbors, svm
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import confusion_matrix
from operator import itemgetter
from itertools import groupby
import numpy as np
import seaborn as sns
import pandas as pd
import random
from seaborn import pairplot
from collections import Counter
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import TomekLinks, EditedNearestNeighbours
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.ensemble import BalancedRandomForestClassifier
import sklearn.metrics as metrics
from scipy.stats import randint as sp_randint
from imblearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def plot_roc(fpr, tpr, roc_auc, clf_name):
    plt.figure()
    plt.title('{} - Receiver Operating Characteristic'.format(clf_name))
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig('2_{}_ROC.png'.format(clf_name), bbox_inches='tight')
    # plt.show()


def string_to_timestamp(date_string):  # convert time string to float value
    time_stamp = time.strptime(date_string, '%Y-%m-%d %H:%M:%S')
    return time.mktime(time_stamp)


def visualise_heatmap(data, labels):
    to_plot = [data[ind] for ind, y in enumerate(labels) if y]
    print(len(to_plot))
    to_plot += random.sample([data[ind] for ind, y in enumerate(labels) if not y], len(to_plot))
    print(len(to_plot))
    df = pd.DataFrame(to_plot)
    df.columns = ['issuercountry', 'txvariantcode', 'issuer_id', 'amount', 'currencycode',
                  'shoppercountry', 'interaction', 'verification', 'cvcresponse', 'creationdate_stamp',
                  'accountcode', 'mail_id', 'ip_id', 'card_id']
    df.drop('creationdate_stamp', axis=1, inplace=True)
    ax = sns.heatmap(df)
    plt.show()


def visualise_pair_plot(data, labels):
    df = pd.DataFrame(data)
    cols = ['issuercountry', 'txvariantcode', 'issuer_id', 'amount', 'currencycode',
            'shoppercountry', 'interaction', 'verification', 'cvcresponse',
            'accountcode', 'mail_id', 'ip_id', 'card_id']

    df.columns = ['issuercountry', 'txvariantcode', 'issuer_id', 'amount', 'currencycode',
                  'shoppercountry', 'interaction', 'verification', 'cvcresponse', 'creationdate_stamp',
                  'accountcode', 'mail_id', 'ip_id', 'card_id']
    df.drop('creationdate_stamp', axis=1, inplace=True)
    df['label'] = labels
    ax = pairplot(df, vars=cols, hue='label', markers=['.', 'x'])
    plt.show()


# In[9]:


def aggregate(before_aggregate, aggregate_feature):
    if aggregate_feature == 'day':
        after_aggregate = []
        pos_date = -1
        before_aggregate.sort(key=itemgetter(9))  # sort by timestamp
        temp = groupby(before_aggregate, itemgetter(-2))
        group_unit = []
        mean = []
        for i, item in temp:  # i is group id
            for jtem in item:  # unit in each group
                group_unit.append(jtem)
            # for feature_i in xrange(6):
            #    mean.append(zip(group_unit)[feature_i])
            # after_aggregate.append(group_unit)
            after_aggregate.append(mean)
            group_unit = []
        # print after_aggregate[0]
        # print before_aggregate[0]
    if aggregate_feature == 'client':
        after_aggregate = []
        pos_client = -3
        before_aggregate.sort(
            key=itemgetter(pos_client))  # sort with cardID firstly，if sort with 2 feature, itemgetter(num1,num2)
        temp = groupby(before_aggregate, itemgetter(pos_client))  # group
        group_unit = []
        for i, item in temp:  # i is group id
            for jtem in item:  # unit in each group
                group_unit.append(jtem)
            after_aggregate.append(group_unit)
            group_unit = []
    return after_aggregate


# In[10]:


def aggregate_mean(before_aggregate):
    # print before_aggregate[0]
    if True:
        after_aggregate = []
        pos_date = -1
        before_aggregate.sort(key=itemgetter(-1))  # sort by timestamp
        temp = groupby(before_aggregate, itemgetter(-1))
        group_unit = []
        mean = []
        for i, item in temp:  # i is group id
            for jtem in item:  # unit in each group
                group_unit.append(list(jtem))
            # print group_unit
            if len(zip(group_unit)) < 2:
                after_aggregate.append(group_unit)
                group_unit = []
            if len(zip(group_unit)) >= 2:
                # print zip(group_unit)
                for feature_i in range(14):
                    # print zip(group_unit)[feature_i]
                    mean.append(sum(zip(*group_unit)[feature_i]) / len(zip(group_unit)))
                after_aggregate.append(mean)
                group_unit = []
                mean = []
        # print after_aggregate[0]
        # print before_aggregate[0]
    return after_aggregate


# In[16]:
def plot_decision_tree(clf):
    from sklearn.externals.six import StringIO
    from IPython.display import Image
    from sklearn.tree import export_graphviz
    import pydotplus
    features = np.array(['txid', 'bookingdate', 'issuercountrycode', 'txvariantcode', 'bin', 'amount', 'currencycode',
                         'shoppercountrycode', 'shopperinteraction', 'simple_journal', 'cardverificationcodesupplied',
                         'cvcresponsecode', 'creationdate', 'accountcode', 'mail_id', 'ip_id', 'card_id'])
    fearure_nums = [0, 4, 8, 10, 12]
    # dot_data = StringIO()
    export_graphviz(clf, out_file='tree.dot', max_depth=5, feature_names=features[fearure_nums],
                    class_names=['non-fraudulent', 'fraudulent'],
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
        # clf = neighbors.KNeighborsClassifier(algorithm='kd_tree', weights='distance', n_jobs=-1)  # nice results
        # clf = LogisticRegression()
        # clf = AdaBoostClassifier(n_estimators=100)
        # clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
        # clf = neighbors.KNeighborsClassifier(algorithm='kd_tree', n_jobs=-1) #good results
        # clf = neighbors.KNeighborsClassifier(algorithm='auto', weights='distance', n_jobs=-1) #nice results
        # clf = BalancedRandomForestClassifier(n_estimators=100, n_jobs=-1)
        if normalize:
            scaler = StandardScaler().fit(x_train)
            x_train = scaler.transform(x_train)
            x_test = scaler.transform(x_test)
        # clf = svm.SVC(kernel='linear')

        clf.fit(x_train, y_train)

        if i == 0 and clf_name == 'DecisionTreeClassifier':
            plot_decision_tree(clf)
        y_predict = clf.predict(x_test)

        # plot roc curve
        total_y_test += list(y_test)
        total_y_prob += list(clf.predict_proba(x_test)[:, 1])

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
        # print('TP: ' + str(TP))
        # print('FP: ' + str(FP))
        # print('FN: ' + str(FN))
        # print('TN: ' + str(TN))
        # print(TP + TN + FP + FN)
        totalFN += FN
        totalFP += FP
        totalTN += TN
        totalTP += TP
        # print confusion_matrix(y_test, answear) watch out the element in confusion matrix
        precision, recall, thresholds = precision_recall_curve(y_test, y_predict)
        predict_proba = clf.predict_proba(x_test)  # the probability of each smple labelled to positive or negative

    print('TOTAL TP: ' + str(totalTP))
    print('TOTAL FP: ' + str(totalFP))
    print('TOTAL FN: ' + str(totalFN))
    print('TOTAL TN: ' + str(totalTN))

    total_y_test = np.array(total_y_test)
    total_y_pred = np.array(total_y_prob)
    fpr, tpr, threshold = metrics.roc_curve(total_y_test, total_y_pred)
    roc_auc = metrics.auc(fpr, tpr)
    plot_roc(fpr, tpr, roc_auc, clf_name)


if __name__ == "__main__":
    filename = 'original_data.csv'
    data = pd.read_csv(filename)
    x = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    # raw_data = open(filename, 'rt')
    # data = np.loadtxt(raw_data, skiprows=1, delimiter=",", dtype=np.float64)
    # x = data[:, :-1]
    # y = data[:, -1]

    # x_array = np.delete(x_array, [0, 1, 2, 3, 5, 6, 7, 9, 11, 13], 1)
    x = np.delete(x, [1, 2, 3, 5, 6, 7, 9, 11, 13], 1)

    # pca = PCA(n_components='mle', svd_solver='full')
    # pca.fit(x_array)
    # x_array = pca.transform(x_array)
    clfs = {'DecisionTreeClassifier': DecisionTreeClassifier(),
            'AdaBoostClassifier': AdaBoostClassifier(n_estimators=100)}
    # 'LogisticRegression': LogisticRegression(solver='newton-cg'),
    # 'RandomForestClassifier': RandomForestClassifier(n_estimators=100, n_jobs=-1)}
    for clf_name, clf in clfs.items():
        usx = np.copy(x)
        usy = np.copy(y)
        if clf_name == 'LogisticRegression':
            make_clf(usx, usy, clf, clf_name, normalize=True)
        else:
            make_clf(usx, usy, clf, clf_name)
