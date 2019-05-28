import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from saxpy.sax import sax_by_chunking
from saxpy.znorm import znorm
from math import ceil
from nltk import ngrams
import warnings
warnings.filterwarnings("ignore")


def add_labels(df, test=True):
    df.columns = df.columns.str.lstrip()
    if test:
        df['ATT_FLAG'] = 0
        df.loc['2017-01-16 09': '2017-01-19 06', 'ATT_FLAG'] = 1
        df.loc['2017-01-30 08':'2017-02-02 00',  'ATT_FLAG'] = 1
        df.loc['2017-02-09 03': '2017-02-10 09', 'ATT_FLAG'] = 1
        df.loc['2017-02-12 01': '2017-02-13 07', 'ATT_FLAG'] = 1
        df.loc['2017-02-24 05': '2017-02-28 08', 'ATT_FLAG'] = 1
        df.loc['2017-03-10 14': '2017-03-13 21', 'ATT_FLAG'] = 1
        df.loc['2017-03-25 20': '2017-03-27 01', 'ATT_FLAG'] = 1
        y = df['ATT_FLAG']  # separate the target values
        df.drop(['ATT_FLAG'], axis=1, inplace=True)
    else:
        df.loc['2016-09-13 23': '2016-09-16 00', 'ATT_FLAG'] = 1
        df.loc['2016-09-26 11': '2016-09-27 10', 'ATT_FLAG'] = 1
        df.loc['2016-10-09 16': '2016-10-11 20', 'ATT_FLAG'] = 1
        df.loc['2016-10-29 19': '2016-11-02 16', 'ATT_FLAG'] = 1
        df.loc['2016-11-26 17': '2016-11-29 04', 'ATT_FLAG'] = 1
        df.loc['2016-12-06 07': '2016-12-10 04', 'ATT_FLAG'] = 1
        df.loc['2016-12-14 15': '2016-12-19 04', 'ATT_FLAG'] = 1
        y = df['ATT_FLAG']
        df.drop(['ATT_FLAG'], axis=1, inplace=True)
    return df, y


def confusion_results(predicted, true):
    TP, FP, TN, FN = 0, 0, 0, 0
    for p, t in zip(predicted, true):
        if p and t:
            TP += 1
        elif p and t!=1:
            FP += 1
        elif not p and t:
            FN += 1
        else:
            TN += 1
    return TP, FP, TN, FN


def uncompress_labels(compressed_data, n, w):
    j = 0
    step = ceil(n/w)
    uncompressed = []
    for i in range(len(compressed_data)):
        uncompressed += [compressed_data[i]] * min(step, n - j)
        j += step
    return uncompressed


def discretize_data(data, w, features, start_date, end_date, dataset, plotting):
    alphabet = 5
    symbol_to_number = {'a': -1, 'b': -0.5, 'c': 0, 'd': 0.5, 'e': 1}
    sax_seqs = {}
    for feature in features:
        print('------------------------------------- Discretizing %s -------------------------------------' % feature)
        sax_str = sax_by_chunking(np.array(data[feature]), w, alphabet)
        sax_seqs[feature] = sax_str

        if plotting:
            normalized_signal = znorm(np.array(data[feature]))
            discrete = uncompress_labels(sax_str, len(normalized_signal), w)
            discrete = [symbol_to_number[d] for d in discrete]

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(pd.DataFrame(normalized_signal, index=data.index).loc[start_date:end_date],
                    label='normalized signal')
            ax.plot(pd.DataFrame(discrete, index=data.index).loc[start_date:end_date], label='discretized signal')
            ax.xaxis.set_major_locator(mdates.DayLocator([5, 10, 15, 20, 25, 30]))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
            plt.xlabel("time")
            plt.ylabel(feature)
            plt.xticks(rotation=45)
            plt.yticks(np.array(list(symbol_to_number.values())), tuple(symbol_to_number.keys()))
            plt.grid()
            plt.legend()
            plt.savefig('plots/sax/%s_discretization_%s.png' % (dataset, feature), bbox_inches='tight')
    return sax_seqs


def train_ngram(dsc_train, w):
    # states object
    states = {}
    sequences = {}

    # get the ngrams for the training data
    all_ngrams = ngrams(list(dsc_train), w)
    cnt = 0.0
    prev = ""
    prev_key = ""
    seq = 0
    try:
        while True:
            # get the next gram
            gram = next(all_ngrams)

            # create key string
            curr = ''.join(gram)
            key = prev + '-' + curr

            # check if the transition already existed
            states[key] = states[key] + 1 if key in states.keys() else 1

            # check if greater subsequence
            if prev_key != key:
                if prev_key != '':
                    if prev_key not in sequences.keys():
                        sequences[prev_key] = seq
                    else:
                        if sequences[prev_key] < seq:
                            sequences[prev_key] = seq
                seq = 1
            else:
                seq += 1

            cnt += 1
            # store the old key
            prev = curr
            prev_key = key
    except StopIteration:
        pass

    for key in states.keys():
        states[key] = states[key] / cnt

    return states, sequences


def test_ngram(dsc_test, w, threshold, states, sequences):
    test_grams = ngrams(list(dsc_test), w)
    prev = ""
    anomalies = []
    prev_key = ""
    seq = 0

    try:
        while True:
            # get the next gram
            gram = next(test_grams)

            # create key string
            curr = ''.join(gram)
            key = prev + '-' + curr

            # check the odds of this transition
            if prev == "":
                anomalies += [0]
            elif key not in states.keys():
                anomalies += [1]
            elif states[key] < threshold:
                anomalies += [1]
            elif seq > sequences[key]:
                anomalies += [1]
            else:
                anomalies += [0]

            # update subsequence
            seq = 1 if prev_key != key else seq + 1

            # store the old key
            prev = curr
            prev_key = key
    except StopIteration:
        pass

    return anomalies


def plot_anomalies(true_anomalies, predicted_anomalies, sensor):
    fig, ax = plt.subplots()
    ax.fill_between(true_anomalies.index, true_anomalies, label='True', alpha=0.5)
    ax.fill_between(true_anomalies.index, predicted_anomalies, label='Predicted', alpha=0.5)
    ax.xaxis.set_major_locator(mdates.DayLocator([5, 10, 15, 20, 25, 30]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
    ax.set_ylabel('Attacks')
    ax.grid()
    plt.legend()
    plt.title('Attacks for sensor {}'.format(sensor))
    plt.xticks(rotation=45)
    plt.xlabel("Date")
    plt.savefig('plots/ngrams/attacks_%s.png' % sensor, bbox_inches='tight')


if __name__ == '__main__':
    print('Reading datasets...')
    train_df1 = pd.read_csv('BATADAL_datasets/BATADAL_training_dataset1.csv', index_col=0, parse_dates=[0],
                            date_parser=lambda x: pd.to_datetime(x, format="%d/%m/%y %H"))

    train_df2 = pd.read_csv('BATADAL_datasets/BATADAL_training_dataset2.csv', index_col=0, parse_dates=[0],
                            date_parser=lambda x: pd.to_datetime(x, format="%d/%m/%y %H"))
    train_df2, train2_anomalies = add_labels(train_df2, test=False)

    test_df = pd.read_csv('BATADAL_datasets/BATADAL_test_dataset.csv', index_col=0, parse_dates=[0],
                          date_parser=lambda x: pd.to_datetime(x, format="%d/%m/%y %H"))
    test_df, true_anomalies = add_labels(test_df)

    print('Discretizing the data with SAX...')
    # ['L_T1', 'L_T3', 'L_T4', 'L_T5', 'L_T6', 'L_T7', 'F_PU1', 'F_PU4', 'F_PU10', 'P_J14', 'P_J422']
    dsc_sensors = discretize_data(train_df1, 1000, ['L_T1', 'F_PU4', 'P_J14'], '2014-09-01', '2014-09-30', 'train1',
                                  False)
    dsc_sensors_tr2 = discretize_data(train_df2, 1000, ['L_T1', 'F_PU4', 'P_J14'], '2016-07-04', '2016-07-30', 'train2',
                                      False)
    dsc_sensors_tst = discretize_data(test_df, 1000, ['L_T1', 'F_PU4', 'P_J14'], '2017-01-04', '2017-01-30', 'test',
                                      False)

    print('N-gram calculation...')
    ngram_results = {}
    for sensor in dsc_sensors.keys():
        print('------------------------------------- Training on %s -------------------------------------' % sensor)
        states, sequences = train_ngram(dsc_sensors[sensor], 2)
        threshold = 1.01
        max_Scm = 0
        for t in [1e-3, 1e-4, 1e-5, 1e-6]:
            trained_anomalies = test_ngram(dsc_sensors_tr2[sensor], 2, t, states, sequences)
            uncompressed_anomalies = uncompress_labels(trained_anomalies, len(train2_anomalies), len(trained_anomalies))
            TP, FP, TN, FN = confusion_results(uncompressed_anomalies, list(train2_anomalies))
            print('TP: %d, FP: %d, TN: %d, FN: %d' % (TP, FP, TN, FN))
            if (TP/(TP+FN) + TN/(TN+FP))/2 > max_Scm:
                threshold = t
        anomalies = test_ngram(dsc_sensors_tst[sensor], 2, threshold, states, sequences)
        ngram_results[sensor] = uncompress_labels(anomalies, len(true_anomalies), len(anomalies))
        plot_anomalies(true_anomalies, ngram_results[sensor], sensor)
