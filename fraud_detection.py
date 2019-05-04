# coding: utf-8

# In[7]:


import datetime
import time
import matplotlib.pyplot as plt
from sklearn import neighbors, svm
from sklearn.decomposition import PCA
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

# In[8]:
from sklearn.preprocessing import StandardScaler


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


if __name__ == "__main__":
    src = 'data_for_student_case.csv'
    ah = open(src, 'r')
    x = []  # contains features
    y = []  # contains labels
    data = []
    color = []
    coversion_dict = {'SEK': 0.09703, 'MXN': 0.04358, 'AUD': 0.63161, 'NZD': 0.58377, 'GBP': 1.13355}
    (issuercountry_set, txvariantcode_set, currencycode_set, shoppercountry_set, interaction_set,
     verification_set, accountcode_set, mail_id_set, ip_id_set, card_id_set) = [set() for _ in range(10)]
    (issuercountry_dict, txvariantcode_dict, currencycode_dict, shoppercountry_dict, interaction_dict,
     verification_dict, accountcode_dict, mail_id_dict, ip_id_dict, card_id_dict) = [{} for _ in range(10)]
    # label_set
    # cvcresponse_set = set()
    ah.readline()  # skip first line
    for line_ah in ah:
        if line_ah.strip().split(',')[
            9] == 'Refused':  # remove the row with 'refused' label, since it's uncertain about fraud
            continue
        if 'na' in str(line_ah.strip().split(',')[14]).lower() or 'na' in str(line_ah.strip().split(',')[4].lower()):
            continue
        bookingdate = string_to_timestamp(line_ah.strip().split(',')[1])  # date reported flaud
        issuercountry = line_ah.strip().split(',')[2]  # country code
        issuercountry_set.add(issuercountry)
        txvariantcode = line_ah.strip().split(',')[3]  # type of card: visa/master
        txvariantcode_set.add(txvariantcode)
        issuer_id = float(line_ah.strip().split(',')[4])  # bin card issuer identifier
        amount = float(line_ah.strip().split(',')[5])  # transaction amount in minor units
        currencycode = line_ah.strip().split(',')[6]
        currencycode_set.add(currencycode)
        amount = coversion_dict[currencycode] * amount  # currency conversion
        shoppercountry = line_ah.strip().split(',')[7]  # country code
        shoppercountry_set.add(shoppercountry)
        interaction = line_ah.strip().split(',')[8]  # online transaction or subscription
        interaction_set.add(interaction)
        if line_ah.strip().split(',')[9] == 'Chargeback':
            label = 1  # label fraud
        else:
            label = 0  # label save
        verification = line_ah.strip().split(',')[10]  # shopper provide CVC code or not
        verification_set.add(verification)
        cvcresponse = int(line_ah.strip().split(',')[11])  # 0 = Unknown, 1=Match, 2=No Match, 3-6=Not checked
        if cvcresponse > 2:
            cvcresponse = 3
        year_info = datetime.datetime.strptime(line_ah.strip().split(',')[12], '%Y-%m-%d %H:%M:%S').year
        month_info = datetime.datetime.strptime(line_ah.strip().split(',')[12], '%Y-%m-%d %H:%M:%S').month
        day_info = datetime.datetime.strptime(line_ah.strip().split(',')[12], '%Y-%m-%d %H:%M:%S').day
        creationdate = str(year_info) + '-' + str(month_info) + '-' + str(day_info)  # Date of transaction
        creationdate_stamp = string_to_timestamp(line_ah.strip().split(',')[12])  # Date of transaction-time stamp
        accountcode = line_ah.strip().split(',')[13]  # merchant’s webshop
        accountcode_set.add(accountcode)
        mail_id = int(float(line_ah.strip().split(',')[14].replace('email', '')))  # mail
        mail_id_set.add(mail_id)
        ip_id = int(float(line_ah.strip().split(',')[15].replace('ip', '')))  # ip
        ip_id_set.add(ip_id)
        card_id = int(float(line_ah.strip().split(',')[16].replace('card', '')))  # card
        card_id_set.add(card_id)
        data.append([issuercountry, txvariantcode, issuer_id, amount, currencycode,
                     shoppercountry, interaction, verification, cvcresponse, creationdate_stamp,
                     accountcode, mail_id, ip_id, card_id, label, creationdate])  # add the interested features here
        # y.append(label)# add the labels
    data = sorted(data, key=lambda k: k[-1])
    day_aggregate = aggregate(data, 'day')
    client_aggregate = aggregate(data, 'client')
    # transaction_num_day = []
    # for item in day_aggregate:
    #     transaction_num_day.append(len(item))
    # plt.figure(1)
    # plt.plot(transaction_num_day, color = 'c', linewidth = 2)
    # plt.plot()
    # plt.text(2,0.0,'Date: 2015-10-8')
    # plt.xlabel('Date')
    # plt.ylabel('Number of Transactions')
    # plt.xlim([0,125])
    # plt.axis('tight')
    # plt.savefig('Day Aggregating.png')
    # transaction_num_client = []
    # for item in client_aggregate:
    #     transaction_num_client.append(len(item))
    # plt.figure(2)
    # plt.plot(transaction_num_client, color = 'c', linewidth = 2)
    # plt.text(99,9668,'Date: 2015-10-8')
    # plt.xlabel('Client ID')
    # plt.ylabel('Number of Transactions')
    # plt.axis('tight')
    # plt.savefig('Client Aggregating.png')

# In[17]:


for item in data:  # split data into x,y
    x.append(item[0:-2])
    y.append(item[-2])
'''map number to each categorial feature'''
for item in list(issuercountry_set):
    issuercountry_dict[item] = list(issuercountry_set).index(item)
for item in list(txvariantcode_set):
    txvariantcode_dict[item] = list(txvariantcode_set).index(item)
print(currencycode_set)
for item in list(currencycode_set):
    currencycode_dict[item] = list(currencycode_set).index(item)
for item in list(shoppercountry_set):
    shoppercountry_dict[item] = list(shoppercountry_set).index(item)
for item in list(interaction_set):
    interaction_dict[item] = list(interaction_set).index(item)
for item in list(verification_set):
    verification_dict[item] = list(verification_set).index(item)
for item in list(accountcode_set):
    accountcode_dict[item] = list(accountcode_set).index(item)
print(len(list(card_id_set)))
# for item in list(card_id_set):
#    card_id_dict[item] = list(card_id_set).index(item)
'''modify categorial feature to number in data set'''
for item in x:
    item[0] = issuercountry_dict[item[0]]
    item[1] = txvariantcode_dict[item[1]]
    item[4] = currencycode_dict[item[4]]
    item[5] = shoppercountry_dict[item[5]]
    item[6] = interaction_dict[item[6]]
    item[7] = verification_dict[item[7]]
    item[10] = accountcode_dict[item[10]]

# x_mean = []
# x_mean = aggregate_mean(x)
# visualise_heatmap(x, y)
# visualise_pair_plot(x, y)
x_mean = x
des = 'original_data.csv'
des1 = 'aggregate_data.csv'
ch_dfa = open(des, 'w')

ch_dfa.write(
    'issuercountry, txvariantcode, issuer_id, amount, currencycode,shoppercountry, interaction, verification, cvcresponse, creationdate_stamp, accountcode, mail_id, ip_id, card_id, label')
ch_dfa.write('\n')

# ch_dfa.write('txid,bookingdate,issuercountrycode,txvariantcode,bin,amount,'+
#             'currencycode,shoppercountrycode,shopperinteraction,simple_journal,'+
#             'cardverificationcodesupplied,cvcresponsecode,creationdate,accountcode,mail_id,ip_id,card_id')
# ch_dfa.write('\n')

sentence = []
for i in range(len(x_mean)):
    for j in range(len(x_mean[i])):
        sentence.append(str(x_mean[i][j]))
    sentence.append(str(y[i]))
    ch_dfa.write(','.join(sentence))
    ch_dfa.write('\n')
    sentence = []
    ch_dfa.flush()

x_array = np.array(x)
x_array = np.delete(x_array, [0, 1, 2, 3, 5, 6, 7, 9, 11, 13], 1)
y_array = np.array(y)

# pca = PCA(n_components='mle', svd_solver='full')
# pca.fit(x_array)
# x_array = pca.transform(x_array)

# tune_clf(x_array, y_array)

usx = x_array
usy = y_array
usx = usx.astype(np.float64)
usy = usy.astype(np.float64)
# x_train, x_test, y_train, y_test = train_test_split(usx, usy, test_size=0.2)
totalTP, totalFP, totalFN, totalTN = 0, 0, 0, 0
total_y_test = []
total_y_pred = []
skf = StratifiedKFold(n_splits=10)
for train_index, test_index in skf.split(usx, usy):
    x_train, x_test = usx[train_index], usx[test_index]
    y_train, y_test = usy[train_index], usy[test_index]

    # test_size: proportion of train/test data
    # sm = SMOTE(k_neighbors=100, n_jobs=-1)
    # x_train, y_train = sm.fit_resample(x_train, y_train)
    # print('Resampled dataset shape %s' % Counter(y_train))

    # ad = ADASYN(n_neighbors=50, n_jobs=-1)
    # x_train, y_train = ad.fit_resample(x_train, y_train)
    # print('Resampled dataset shape %s' % Counter(y_train))

    # en = EditedNearestNeighbours(n_neighbors=3, n_jobs=-1)
    # x_train, y_train = en.fit_resample(x_train, y_train)
    # print('Resampled dataset shape %s' % Counter(y_train))

    # tl = TomekLinks(sampling_strategy='all', random_state=42, n_jobs=-1)
    # x_train, y_train = tl.fit_resample(x_train, y_train)
    # print('Resampled dataset shape %s' % Counter(y_train))

    # sm = SMOTETomek()
    # x_train, y_train = sm.fit_resample(x_train, y_train)
    # print('Resampled dataset shape %s' % Counter(y_train))

    # clf = neighbors.KNeighborsClassifier(algorithm='kd_tree', n_jobs=-1)
    # clf = neighbors.KNeighborsClassifier(algorithm='auto', weights='distance', n_jobs=-1)
    # clf = RandomForestClassifier( n_estimators=100, n_jobs=-1)
    # clf = BalancedRandomForestClassifier(n_estimators=100, n_jobs=-1)
    # clf = AdaBoostClassifier()
    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    clf = svm.SVC(kernel='linear', class_weight='balanced')


    clf.fit(x_train, y_train)
    y_predict = clf.predict(x_test)

    # plot roc curve
    total_y_test += list(y_test)
    total_y_pred += list(y_predict)

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
total_y_pred = np.array(total_y_pred)
fpr, tpr, threshold = metrics.roc_curve(total_y_test, total_y_pred)
roc_auc = metrics.auc(fpr, tpr)
plot_roc(fpr, tpr, roc_auc)
