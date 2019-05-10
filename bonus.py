# coding: utf-8

import datetime
import pickle
import time
from copy import deepcopy

from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
from collections import Counter
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import EditedNearestNeighbours, OneSidedSelection, TomekLinks
from imblearn.combine import SMOTEENN, SMOTETomek
from sklearn.preprocessing import StandardScaler


def string_to_timestamp(date_string):
    '''
    Function coverting a time string to a float timestamp
    '''
    time_stamp = time.strptime(date_string, '%Y-%m-%d %H:%M:%S')
    return time.mktime(time_stamp)


def create_dataset():
    """
    Create the dataset from the original csv file
    """
    src = 'data_for_student_case.csv'
    ah = open(src, 'r')
    x = []  # contains features
    y = []  # contains labels
    data = []
    color = []
    conversion_dict = {'SEK': 0.09703, 'MXN': 0.04358, 'AUD': 0.63161, 'NZD': 0.58377, 'GBP': 1.13355}
    (issuercountry_set, txvariantcode_set, currencycode_set, shoppercountry_set, interaction_set,
     verification_set, accountcode_set, mail_id_set, ip_id_set, card_id_set) = [set() for _ in range(10)]
    (issuercountry_dict, txvariantcode_dict, currencycode_dict, shoppercountry_dict, interaction_dict,
     verification_dict, accountcode_dict, mail_id_dict, ip_id_dict, card_id_dict) = [{} for _ in range(10)]

    ah.readline()  # skip first line
    for line_ah in ah:
        if line_ah.strip().split(',')[9] == 'Refused':  # remove the row with 'refused' label, since it's uncertain about fraud
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
        amount = conversion_dict[currencycode] * amount  # currency conversion
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
        crd = datetime.datetime.strptime(line_ah.strip().split(',')[12], '%Y-%m-%d %H:%M:%S')
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
                     shoppercountry, interaction, verification, cvcresponse, crd,
                     accountcode, mail_id, ip_id, card_id, label, creationdate])  # add the interested features here

    data = sorted(data, key=lambda k: k[9])  # data sorted according to transaction-time stamp

    for item in data:  # split data into x,y
        x.append(item[0:-2])
        y.append(item[-2])

    # map number to each categorial feature
    for item in list(issuercountry_set):
        issuercountry_dict[item] = list(issuercountry_set).index(item)
    for item in list(txvariantcode_set):
        txvariantcode_dict[item] = list(txvariantcode_set).index(item)
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

    # modify categorial feature to number in data set
    for item in x:
        item[0] = issuercountry_dict[item[0]]
        item[1] = txvariantcode_dict[item[1]]
        item[4] = currencycode_dict[item[4]]
        item[5] = shoppercountry_dict[item[5]]
        item[6] = interaction_dict[item[6]]
        item[7] = verification_dict[item[7]]
        item[10] = accountcode_dict[item[10]]

    # The "original_data_for_aggr.csv" numeric dataset
    # used to compute the aggregated features is created
    des = 'original_data_for_aggr.csv'
    ch_dfa = open(des, 'w')

    ch_dfa.write(
        'issuercountry,txvariantcode,issuer_id,amount,currencycode,shoppercountry,interaction,verification,cvcresponse,creationdate,accountcode,mail_id,ip_id,card_id,label')
    ch_dfa.write('\n')

    sentence = []
    for i in range(len(x)):
        for j in range(len(x[i])):
            sentence.append(str(x[i][j]))
        sentence.append(str(y[i]))
        ch_dfa.write(','.join(sentence))
        ch_dfa.write('\n')
        sentence = []
        ch_dfa.flush()


def make_clf(usx, usy, clf, clf_name, strategy='SMOTE', normalize=False, cutoff=0.5):
    """
    Makes classification with the given parameters and print the results
    :param usx: features
    :param usy: labels
    :param clf: classifier
    :param clf_name: name of the classifier
    :param strategy: sampling strategy to be used
    :param normalize: boolean to decide whether to normilize or not
    :param cutoff: cutoff value for the classification threshold
    :return:
    """
    print('----------{0} with {1}----------'.format(clf_name, strategy))
    totalTP, totalFP, totalFN, totalTN = 0, 0, 0, 0
    total_y_test = []
    total_y_prob = []
    skf = StratifiedKFold(n_splits=10, shuffle=True)
    for train_index, test_index in skf.split(usx, usy):
        x_train, x_test = usx[train_index], usx[test_index]
        y_train, y_test = usy[train_index], usy[test_index]

        # select sampling strategy
        if strategy == 'SMOTE':
            sm = SMOTE(sampling_strategy=0.5, n_jobs=-1)
            x_train, y_train = sm.fit_resample(x_train, y_train)
            print('Resampled dataset shape %s' % Counter(y_train))
        elif strategy == 'ADASYN':
            ad = ADASYN(n_jobs=-1)
            x_train, y_train = ad.fit_resample(x_train, y_train)
            print('Resampled dataset shape %s' % Counter(y_train))
        elif strategy == 'ENN':
            en = EditedNearestNeighbours(n_jobs=-1)
            x_train, y_train = en.fit_resample(x_train, y_train)
            print('Resampled dataset shape %s' % Counter(y_train))
        elif strategy == 'TL':
            tl = TomekLinks(sampling_strategy='auto', n_jobs=-1)
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
        elif strategy == 'SMOTEEN':
            smoteenn = SMOTEENN(smote=SMOTE(sampling_strategy=0.5, n_jobs=-1), enn=EditedNearestNeighbours(n_jobs=-1))
            x_train, y_train = smoteenn.fit_resample(x_train, y_train)
            print('Resampled dataset shape %s' % Counter(y_train))

        # normalize data if needed
        if normalize:
            scaler = StandardScaler().fit(x_train)
            x_train = scaler.transform(x_train)
            x_test = scaler.transform(x_test)

        # fit classifier
        clf.fit(x_train, y_train)

        # predict labels and their probaility
        y_predict = clf.predict(x_test)
        y_predict_proba = clf.predict_proba(x_test)

        total_y_test += list(y_test)
        total_y_prob += list(y_predict_proba[:, 1])

        # modify the threshold for the two classes
        if cutoff < 0.5:
            for i in range(len(y_predict)):
                if y_predict[i] == 0 and y_predict_proba[i, 1] >= cutoff:
                    y_predict[i] = 1
        elif cutoff > 0.5:
            for i in range(len(y_predict)):
                if y_predict[i] == 1 and y_predict_proba[i, 1] <= cutoff:
                    y_predict[i] = 0

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
        totalFN += FN
        totalFP += FP
        totalTN += TN
        totalTP += TP

    print('TOTAL TP: ' + str(totalTP))
    print('TOTAL FP: ' + str(totalFP))
    print('TOTAL FN: ' + str(totalFN))
    print('TOTAL TN: ' + str(totalTN))


# Below there are the functions that are used for the aggregated features
def custom_aggr(cols, d):
    l = 0
    ids = cols['card_id']
    creationdate = cols['creationdate']
    for id, crd in zip(ids, creationdate):
        l += len([v for v in d[id] if 0 <= (crd - v).days <= 30]) - 1
    return l


def custom_aggr_am_per_trans(cols, d_crd, d_am):
    l = 0
    ids = cols['card_id']
    creationdate = cols['creationdate']
    euroamount = cols['EuroAmount']
    for id, crd, em in zip(ids, creationdate, euroamount):
        l += np.mean([v for i, v in enumerate(d_am[id]) if 0 <= (crd - d_crd[id][i]).days <= 30])
    return int(round(l))


def custom_aggr_avg_over_3_mon(cols, d_crd, d_am):
    l = 0
    ids = cols['card_id']
    creationdate = cols['creationdate']
    euroamount = cols['EuroAmount']
    for id, crd, em in zip(ids, creationdate, euroamount):
        l += np.sum([v for i, v in enumerate(d_am[id]) if 0 <= (crd - d_crd[id][i]).days <= 90])
    l = int(round(l / 12))
    return l


def custom_aggr_avg_daily_over_month(cols, d_crd, d_am):
    l = 0
    ids = cols['card_id']
    creationdate = cols['creationdate']
    euroamount = cols['EuroAmount']
    for id, crd, em in zip(ids, creationdate, euroamount):
        l += np.sum([v for i, v in enumerate(d_am[id]) if 0 <= (crd - d_crd[id][i]).days <= 30])
    l = int(round(l / 30))
    return l


def custom_aggr_amount_merchant_over_month(cols, d_crd, d_am):
    l = 0
    card_id = cols['card_id']
    euroamount = cols['EuroAmount']
    accountcode = cols['accountcode']
    creationdate = cols['creationdate']
    for cid, acc, crd, em in zip(card_id, accountcode, creationdate, euroamount):
        l += np.sum([v for i, v in enumerate(d_am[(cid, acc)]) if 0 <= (crd - d_crd[(cid, acc)][i]).days <= 30]) - em
    l = int(round(l / 30))
    return l


def custom_aggr_diff_merchant_over_month(cols, d_crd):
    l = 0
    card_id = cols['card_id']
    euroamount = cols['EuroAmount']
    accountcode = cols['accountcode']
    creationdate = cols['creationdate']
    for cid, acc, crd, em in zip(card_id, accountcode, creationdate, euroamount):
        l += len([v for v in d_crd[(cid, acc)] if 0 <= (crd - v).days <= 30]) - 1
    return int(round(l))


def custom_aggr_amount_merchant_over_3_months(cols, d_crd, d_am):
    l = 0
    card_id = cols['card_id']
    euroamount = cols['EuroAmount']
    accountcode = cols['accountcode']
    creationdate = cols['creationdate']
    for cid, acc, crd, em in zip(card_id, accountcode, creationdate, euroamount):
        l += np.sum([v for i, v in enumerate(d_am[(cid, acc)]) if 0 <= (crd - d_crd[(cid, acc)][i]).days <= 90]) - em
    l = int(round(l / 12))
    return l


def custom_aggr_amount_same_day(cols, d_crd, d_am):
    l = 0
    card_id = cols['card_id']
    euroamount = cols['EuroAmount']
    creationdate = cols['creationdate']
    for cid, crd, em in zip(card_id, creationdate, euroamount):
        l += np.sum([v for i, v in enumerate(d_am[cid]) if crd.date() == d_crd[cid][i].date()]) - em
    return l


def custom_aggr_number_same_day(cols, d_crd, d_am):
    l = 0
    card_id = cols['card_id']
    creationdate = cols['creationdate']
    for cid, crd in zip(card_id, creationdate):
        l += len([v for i, v in enumerate(d_am[cid]) if crd.date() == d_crd[cid][i].date()]) - 1
    return l


def custom_aggr_amount_same_merchant(cols, d_crd, d_am):
    l = 0
    euroamount = cols['EuroAmount']
    accountcode = cols['accountcode']
    creationdate = cols['creationdate']
    d_am = np.array(d_am)
    for acc, crd, em in zip(accountcode, creationdate, euroamount):
        l += np.sum([v for i, v in enumerate(d_am[acc]) if 0 <= (crd - d_crd[acc][i]).days <= 30]) - em
        # l += np.sum([(d_am[list(map(lambda x: 0 <= (crd - x).days <= 30, d_crd[acc]))])]) - em
    l = int(round(l / 30))
    return l


def custom_aggr_num_same_merchant(cols, d_crd):
    l = 0
    euroamount = cols['EuroAmount']
    accountcode = cols['accountcode']
    creationdate = cols['creationdate']
    for acc, crd, em in zip(accountcode, creationdate, euroamount):
        l += len([v for v in d_crd[acc] if 0 <= (crd - v).days <= 30]) - 1
    return int(round(l-1))


def custom_aggr_amount_currency_over_month(cols, d_crd, d_am):
    l = 0
    card_id = cols['card_id']
    euroamount = cols['EuroAmount']
    currencycode = cols['currencycode']
    creationdate = cols['creationdate']
    for cid, cur, crd, em in zip(card_id, currencycode, creationdate, euroamount):
        l += np.sum([v for i, v in enumerate(d_am[(cid, cur)]) if 0 <= (crd - d_crd[(cid, cur)][i]).days <= 30]) - em
    l = int(round(l / 30))
    return l


def custom_aggr_diff_currency_over_month(cols, d_crd):
    l = 0
    card_id = cols['card_id']
    euroamount = cols['EuroAmount']
    currencycode = cols['currencycode']
    creationdate = cols['creationdate']
    for cid, cur, crd, em in zip(card_id, currencycode, creationdate, euroamount):
        l += len([v for v in d_crd[(cid, cur)] if 0 <= (crd - v).days <= 30]) - 1
    return int(round(l))


def custom_aggr_amount_country_over_month(cols, d_crd, d_am):
    l = 0
    card_id = cols['card_id']
    euroamount = cols['EuroAmount']
    shoppercountrycode = cols['shoppercountrycode']
    creationdate = cols['creationdate']
    for cid, shc, crd, em in zip(card_id, shoppercountrycode, creationdate, euroamount):
        l += np.sum([v for i, v in enumerate(d_am[(cid, shc)]) if 0 <= (crd - d_crd[(cid, shc)][i]).days <= 30]) - em
    l = int(round(l / 30))
    return l


def custom_aggr_diff_country_over_month(cols, d_crd):
    l = 0
    card_id = cols['card_id']
    euroamount = cols['EuroAmount']
    shoppercountrycode = cols['shoppercountrycode']
    creationdate = cols['creationdate']
    for cid, shc, crd, em in zip(card_id, shoppercountrycode, creationdate, euroamount):
        l += len([v for v in d_crd[(cid, shc)] if 0 <= (crd - v).days <= 30]) - 1
    return int(round(l))


def custom_aggr_amount_merchant_type_over_3_months(cols, d_crd, d_am):
    l = 0
    euroamount = cols['EuroAmount']
    accountcode = cols['accountcode']
    creationdate = cols['creationdate']
    for acc, crd, em in zip(accountcode, creationdate, euroamount):
        l += np.sum([v for i, v in enumerate(d_am[acc]) if 0 <= (crd - d_crd[acc][i]).days <= 90]) - em
    l = int(round(l / 12))
    return l


def custom_aggr_diff_merchant_type_over_3_months(cols, d_crd):
    l = 0
    accountcode = cols['accountcode']
    creationdate = cols['creationdate']
    for acc, crd in zip(accountcode, creationdate):
        l += len([v for v in d_crd[acc] if 0 <= (crd - v).days <= 90]) - 1
    l = int(round(l))
    return l


def aggregate_data(data):
    """
    Computes the aggregated features
    :param data: original dataset
    :return: dataset with aggregated features
    """
    gen = data[['card_id', 'creationdate', 'EuroAmount', 'txvariantcode', 'accountcode', 'shoppercountrycode', 'currencycode']]

    # number of transactions on the last month
    trans_over_month = data[['card_id', 'creationdate']]
    d1 = gen.groupby('card_id')['creationdate'].apply(list).to_dict()
    trans_over_month = trans_over_month.groupby(['card_id', 'creationdate'], as_index=False)[['card_id', 'creationdate']].agg(custom_aggr, d1).to_frame('trans_past_month')
    data = pd.merge(data, trans_over_month, how='inner', on=['card_id', 'creationdate'])

    # amount spent on the transactions of the last month
    amount_per_trans = data[['card_id', 'EuroAmount', 'creationdate']]
    d2 = gen.groupby('card_id')['EuroAmount'].apply(list).to_dict()
    amount_per_trans = amount_per_trans.groupby(['card_id', 'EuroAmount', 'creationdate'], as_index=False)['card_id', 'EuroAmount', 'creationdate'].agg(custom_aggr_am_per_trans, d1, d2).to_frame('avg_amount_per_trans')
    data = pd.merge(data, amount_per_trans, how='inner', on=['card_id', 'creationdate'])

    # average amount spent over last 3 months
    avg_over_3_mon = data[['card_id', 'EuroAmount', 'creationdate']]
    avg_over_3_mon = avg_over_3_mon.groupby(['card_id', 'EuroAmount', 'creationdate'], as_index=False)[['card_id', 'EuroAmount', 'creationdate']].agg(custom_aggr_avg_over_3_mon, d1, d2).to_frame('avg_over_3_mon')
    data = pd.merge(data, avg_over_3_mon, how='inner', on=['card_id', 'creationdate'])

    # average daily amount spent over last month
    avg_daily_over_month = data[['card_id', 'EuroAmount', 'creationdate']]
    avg_daily_over_month = avg_daily_over_month.groupby(['card_id', 'EuroAmount', 'creationdate'], as_index=False)['card_id', 'EuroAmount', 'creationdate'].agg(custom_aggr_avg_daily_over_month, d1, d2).to_frame('avg_daily_over_month')
    data = pd.merge(data, avg_daily_over_month, how='inner', on=['card_id', 'creationdate'])

    # average daily amount spent over last month
    amount_merchant_over_month = data[['card_id', 'accountcode', 'creationdate','EuroAmount']]
    d3 = gen.groupby(['card_id', 'accountcode'])['creationdate'].apply(list).to_dict()
    d4 = gen.groupby(['card_id', 'accountcode'])['EuroAmount'].apply(list).to_dict()
    amount_merchant_over_month = amount_merchant_over_month.groupby(['card_id', 'accountcode', 'creationdate', 'EuroAmount'], as_index=False)['card_id', 'accountcode', 'creationdate', 'EuroAmount'].agg(custom_aggr_amount_merchant_over_month, d3, d4).to_frame('amount_merchant_over_month')
    data = pd.merge(data, amount_merchant_over_month, how='inner', on=['card_id', 'accountcode', 'creationdate', 'EuroAmount'])

    # number of different merchant over the last month
    diff_merchant_over_month = data[['card_id', 'accountcode', 'creationdate','EuroAmount']]
    diff_merchant_over_month = diff_merchant_over_month.groupby(['card_id', 'accountcode', 'creationdate','EuroAmount'], as_index=False)['card_id', 'accountcode', 'creationdate','EuroAmount'].agg(custom_aggr_diff_merchant_over_month, d3).to_frame('diff_merchant_over_month')
    data = pd.merge(data, diff_merchant_over_month, how='inner', on=['card_id', 'accountcode', 'creationdate','EuroAmount'])

    # amount spent the last 3 month one this merchant
    amount_merchant_over_3_months = data[['card_id', 'EuroAmount', 'accountcode', 'creationdate']]
    amount_merchant_over_3_months = \
    amount_merchant_over_3_months.groupby(['card_id', 'EuroAmount', 'accountcode', 'creationdate'], as_index=False).agg(custom_aggr_amount_merchant_over_3_months, d3, d4).to_frame('amount_merchant_over_3_months')
    data = pd.merge(data, amount_merchant_over_3_months, how='inner', on=['card_id', 'EuroAmount', 'accountcode', 'creationdate'])

    # amount spent by this card the same day as the one of the transaction
    amount_same_day = data[['card_id', 'creationdate', 'EuroAmount']]
    amount_same_day = amount_same_day.groupby(['card_id', 'creationdate', 'EuroAmount'], as_index=False)['card_id', 'creationdate', 'EuroAmount'].agg(custom_aggr_amount_same_day,d1, d2).to_frame('amount_same_day')
    data = pd.merge(data, amount_same_day, how='inner', on=['card_id', 'creationdate', 'EuroAmount'])

    # number of trancactions done the same day by this card
    number_same_day = data[['card_id', 'creationdate']]
    number_same_day = number_same_day.groupby(['card_id', 'creationdate'], as_index=False)['card_id', 'creationdate'].agg(custom_aggr_number_same_day, d1, d2).to_frame('number_same_day')
    data = pd.merge(data, number_same_day, how='inner', on=['card_id', 'creationdate'])

    # amount spent on this currency over last month by this card
    amount_currency_over_month = data[['card_id', 'currencycode', 'creationdate', 'EuroAmount']]
    d7 = gen.groupby(['card_id', 'currencycode'])['creationdate'].apply(list).to_dict()
    d8 = gen.groupby(['card_id', 'currencycode'])['EuroAmount'].apply(list).to_dict()
    amount_currency_over_month = amount_currency_over_month.groupby(['card_id', 'currencycode', 'creationdate', 'EuroAmount'], as_index=False)['card_id', 'currencycode', 'creationdate', 'EuroAmount'].agg(custom_aggr_amount_currency_over_month, d7,d8).to_frame('amount_currency_over_month')
    data = pd.merge(data, amount_currency_over_month, how='inner', on=['card_id', 'currencycode', 'creationdate', 'EuroAmount'])


    # different currencies used by this card the last month
    diff_currency_over_month = data[['card_id', 'currencycode', 'creationdate', 'EuroAmount']]
    diff_currency_over_month = diff_currency_over_month.groupby(['card_id', 'currencycode', 'creationdate', 'EuroAmount'], as_index=False)['card_id', 'currencycode', 'creationdate', 'EuroAmount'].agg(custom_aggr_diff_currency_over_month,d7).to_frame('diff_currency_over_month')
    data = pd.merge(data, diff_currency_over_month, how='inner', on=['card_id', 'currencycode', 'creationdate', 'EuroAmount'])


    # amount spent on this country over the last month
    amount_country_over_month = data[['card_id', 'shoppercountrycode', 'creationdate', 'EuroAmount']]
    d9 = gen.groupby(['card_id', 'shoppercountrycode'])['creationdate'].apply(list).to_dict()
    d10 = gen.groupby(['card_id', 'shoppercountrycode'])['EuroAmount'].apply(list).to_dict()
    amount_country_over_month = amount_country_over_month.groupby(['card_id', 'shoppercountrycode', 'creationdate', 'EuroAmount'],as_index=False)['card_id', 'shoppercountrycode', 'creationdate', 'EuroAmount'].agg(custom_aggr_amount_country_over_month,d9, d10).to_frame('amount_country_over_month')
    data = pd.merge(data, amount_country_over_month, how='inner', on=['card_id', 'shoppercountrycode', 'creationdate', 'EuroAmount'])


    # number of different countries of transactions the last month
    diff_country_over_month = data[['card_id', 'shoppercountrycode', 'creationdate', 'EuroAmount']]
    diff_country_over_month = diff_country_over_month.groupby(['card_id', 'shoppercountrycode', 'creationdate', 'EuroAmount'], as_index=False)['card_id', 'shoppercountrycode', 'creationdate', 'EuroAmount'].agg(custom_aggr_diff_country_over_month,d9).to_frame('diff_country_over_month')
    data = pd.merge(data, diff_country_over_month, how='inner', on=['card_id', 'shoppercountrycode', 'creationdate', 'EuroAmount'])

    return data


def data_encoding(data, column_name, threshold):
    count = dict(data[column_name].value_counts())
    mapping = {}
    for id in count.keys():
        if count[id] > threshold:
            mapping[id] = id
        else:
            mapping[id] = 'dk'
    data[column_name] = data[column_name].map(mapping)
    return data


if __name__ == "__main__":
    # uncomment the following lines to create the dataset. I will take some time !!!
    # create_dataset()
    # filename = 'original_data_for_aggr.csv'
    # data = pd.read_csv(filename)
    # data = data.rename(columns={'amount': 'EuroAmount', 'shoppercountry': 'shoppercountrycode'})
    #
    # # add aggregated features
    # data = aggregate_data(deepcopy(data))

    # use this to load the aggregated features instead of calculating them again
    with open('data_numerical.pickle', 'rb') as handle:
        data = pickle.load(handle)

    # keep only useful features
    data = data.drop(['txvariantcode', 'EuroAmount', 'interaction','verification', 'creationdate', 'mail_id'], axis=1)

    # convert bin to integer
    data['bin'] = data.apply(lambda r: int(round(r['issuer_id'])), axis=1)


    # encode data
    data = data_encoding(deepcopy(data), 'issuer_id', 3)
    data = data_encoding(deepcopy(data), 'card_id', 10)

    # one-hot encoding
    data = pd.get_dummies(data)

    # get features and labels
    x = data.drop('label', axis=1).values
    y = data['label'].values


    # apply pca
    print('Applying PCA')
    pca = PCA(n_components=100)
    x = pca.fit_transform(x)
    print('PCA done')

    # define the classifiers used and their hyperparameters
    clfs = {'RandomForestClassifier': RandomForestClassifier(n_estimators=50, criterion='entropy',
                                                             class_weight='balanced', n_jobs=-1),
            'DecisionTreeClassifier': DecisionTreeClassifier(criterion='entropy', class_weight='balanced')}

    # make classification
    for clf_name, clf in clfs.items():
        usx = np.copy(x)
        usy = np.copy(y)
        make_clf(usx, usy, clf, clf_name, strategy='SMOTE', cutoff=0.4)
