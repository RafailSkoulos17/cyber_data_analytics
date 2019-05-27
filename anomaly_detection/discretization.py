import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from saxpy.sax import sax_by_chunking
from saxpy.znorm import znorm
from math import ceil
from nltk import ngrams
from statistics import mode


def uncompress_labels(compressed_data, n, w):
    j = 0
    step = ceil(n/w)
    uncompressed = []
    for i in range(len(compressed_data)):
        uncompressed += [compressed_data[i]] * min(step, n - j)
        j += step
    return uncompressed


def compress_labels(data, w):
    compressed = []
    step = ceil(len(data)/w)
    for i in range(0, len(data), step):
        compressed += [mode(data[i:min(len(data), i+step)])]
    return compressed  # TODO: check if the final length is correct


def discretize_data(data, w, features, start_date, end_date, dataset, plotting):
    alphabet = 3
    symbol_to_number = {'a': -1, 'b': 0, 'c': 1}
    sax_seqs = {}
    for feature in features:
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
            plt.yticks(np.array([-1, 0, 1]), ('a', 'b', 'c'))
            plt.grid()
            plt.legend()
            plt.savefig('plots/sax/%s_discretization_%s.png' % (dataset, feature), bbox_inches='tight')
    return sax_seqs


def train_ngram(dsc_train, w):
    # states object
    states = {}

    # get the ngrams for the training data
    all_ngrams = ngrams(list(dsc_train), w)
    cnt = 0.0
    prev = ""
    try:
        while True:
            # get the next gram
            gram = all_ngrams.next()

            # create key string
            curr = ''.join(gram)
            key = prev + '-' + curr

            # check if the transition already existed
            states[key] = states[key] + 1 if key in states else 1
            cnt += 1

            # store the old key
            prev = curr
    except StopIteration:
        pass

    for key in states:
        states[key] = states[key] / cnt

    return states


def test_ngram(dsc_test, w, threshold, states):
    test_grams = ngrams(list(dsc_test), w)
    prev = ""
    anomalies = []

    try:
        while True:
            # get the next gram
            gram = test_grams.next()

            # create key string
            curr = ''.join(gram)
            key = prev + '-' + curr

            # check the odds of this transition
            if prev == "":
                anomalies += [0]
            elif key not in states:
                anomalies += [1]
            elif states[key] < threshold:
                anomalies += [1]
            else:
                anomalies += [0]

            # store the old key
            prev = curr
    except StopIteration:
        pass

    return anomalies


if __name__ == '__main__':
    print('Reading datasets...')
    train_df1 = pd.read_csv('BATADAL_datasets/BATADAL_training_dataset1.csv', index_col=0, parse_dates=[0],
                            date_parser=lambda x: pd.to_datetime(x, format="%d/%m/%y %H"))
    train_df2 = pd.read_csv('BATADAL_datasets/BATADAL_training_dataset2.csv', index_col=0, parse_dates=[0],
                            date_parser=lambda x: pd.to_datetime(x, format="%d/%m/%y %H"))
    y_train2 = list(train_df2['ATT_FLAG'])
    test_df = pd.read_csv('BATADAL_datasets/BATADAL_test_dataset.csv', index_col=0, parse_dates=[0],
                          date_parser=lambda x: pd.to_datetime(x, format="%d/%m/%y %H"))
    y_test = list(test_df['ATT_FLAG'])  # TODO: add the function that puts labels on test data

    print('Discretizing the data with SAX...')
    dsc_sensors = discretize_data(train_df1, 1000, ['L_T1', 'L_T4', 'L_T7', 'F_PU1', 'F_PU10'], '2014-09-01',
                                  '2014-09-30', 'train1', True)
    dsc_sensors_tr2 = discretize_data(train_df2, 1000, ['L_T1', 'L_T4', 'L_T7', 'F_PU1', 'F_PU10'], '2016-01-01',
                                      '2016-01-30', 'train2', True)
    dsc_sensors_tst = discretize_data(test_df, 1000, ['L_T1', 'L_T4', 'L_T7', 'F_PU1', 'F_PU10'], '2017-01-01',
                                      '2017-01-30', 'test', True)

    print('N-gram calculation...')
    ngram_results = {}
    for sensor in dsc_sensors.keys():
        states = train_ngram(dsc_sensors[sensor], 2)
        threshold = min(states.values())
        anomalies = test_ngram(dsc_sensors_tst[sensor], 2, threshold, states)
        ngram_results[sensor] = uncompress_labels(anomalies, len(y_test), len(anomalies))