import pandas as pd
from anomaly_detection.discretization import add_labels, confusion_results
from anomaly_detection.pca.pca_detection import pca_detect
from anomaly_detection.pca.utils import get_score
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings("ignore")


def find_anomaly_blocks(anomaly_labels):
    anomaly_blocks = []
    prev = 0
    start = 0
    for ind, label in enumerate(anomaly_labels):
        if label == 1 and prev != 1:
            start = ind
        elif label != 1 and prev == 1:
            anomaly_blocks += [(start, ind)]
        prev = label
    return anomaly_blocks


def check_first_occurrence(predicted, anomaly_blocks):
    attacks = 0
    time_init = {}
    for ind, p in enumerate(predicted):
        membership = list(map(lambda x: x[0] <= p < x[1], anomaly_blocks))
        if True in membership:
            if membership.index(True) not in time_init.keys():
                time_init[membership.index(True)] = ind
                attacks += 1
    return attacks, time_init


def merge_predictions(predicted):
    final_predictions = np.array([0] * len(list(predicted.values())[0]))
    for pred in predicted.values():
        final_predictions = final_predictions | np.array(pred)
    return final_predictions


def pointwise_precision_recall(predicted, true):
    precision = []
    recall = []
    for i in range(len(predicted)):
        TP, FP, TN, FN = confusion_results(predicted[:i+1], true[:i+1])

        if not TP and not FP:
            precision += [1]
        else:
            precision += [TP/(TP+FP)]

        if not TP and not FN:
            precision += [1]
        else:
            recall += [TP/(TP+FN)]
    return precision, recall


if __name__ == '__main__':
    test_df = pd.read_csv('BATADAL_datasets/BATADAL_test_dataset.csv', index_col=0, parse_dates=[0],
                          date_parser=lambda x: pd.to_datetime(x, format="%d/%m/%y %H"))
    _, true_anomalies = add_labels(test_df)

    with open('discrete_all.pickle', 'rb') as handle:
        ngrams_results = pickle.load(handle)

    pca_results = pca_detect()

    with open('arma_all.pickle', 'rb') as handle:
        arma_results = pickle.load(handle)

    print('------------------------ ARMA Results per sensor ------------------------')
    for sensor in arma_results.keys():
        TP, FP, TN, FN = confusion_results(arma_results[sensor], list(true_anomalies))
        print('-----------------------> %s <-----------------------' % sensor)
        print('True positive: ', TP)
        print('False positive: ', FP)
        print('True negative: ', TN)
        print('False negative: ', FN)
        attacks, _ = check_first_occurrence(arma_results[sensor], find_anomaly_blocks(arma_results[sensor]))
        print('Attacks identified: ', attacks)
        _, _, _, _, _, _, Sttd, Scm, S = get_score([i for i, val in enumerate(arma_results[sensor]) if val == 1],
                                                   [i for i, val in enumerate(list(true_anomalies)) if val == 1],
                                                   list(true_anomalies))
        print('Sttd: ', Sttd)  # TODO: Check Sttd
        print('Scm: ', Scm)
        print('S: ', S)

    print('------------------------ N-grams Results per sensor ------------------------')
    for sensor in ngrams_results.keys():
        TP, FP, TN, FN = confusion_results(ngrams_results[sensor], list(true_anomalies))
        print('-----------------------> %s <-----------------------' % sensor)
        print('True positive: ', TP)
        print('False positive: ', FP)
        print('True negative: ', TN)
        print('False negative: ', FN)
        attacks, _ = check_first_occurrence(ngrams_results[sensor], find_anomaly_blocks(ngrams_results[sensor]))
        print('Attacks identified: ', attacks)
        _, _, _, _, _, _, Sttd, Scm, S = get_score([i for i, val in enumerate(ngrams_results[sensor]) if val == 1],
                                                   [i for i, val in enumerate(list(true_anomalies)) if val == 1],
                                                   list(true_anomalies))
        print('Sttd: ', Sttd)
        print('Scm: ', Scm)
        print('S: ', S)

    arma_merged = merge_predictions(arma_results)
    tp_arma, fp_arma, fn_arma, tn_arma, _, _, Sttd_arma, Scm_arma, S_arma = get_score(
        [i for i, val in enumerate(arma_merged) if val == 1],
        [i for i, val in enumerate(list(true_anomalies)) if val == 1],
        list(true_anomalies))

    ngrams_merged = merge_predictions(ngrams_results)
    tp_ngrams, fp_ngrams, fn_ngrams, tn_ngrams, _, _, Sttd_ngrams, Scm_ngrams, S_ngrams = get_score(
        [i for i, val in enumerate(ngrams_merged) if val == 1],
        [i for i, val in enumerate(list(true_anomalies)) if val == 1],
        list(true_anomalies))

    tp_pca, fp_pca, fn_pca, tn_pca, _, _, Sttd_pca, Scm_pca, S_pca = get_score(
        [i for i, val in enumerate(pca_results) if val == 1],
        [i for i, val in enumerate(list(true_anomalies)) if val == 1],
        list(true_anomalies))

    attacks_arma, _ = check_first_occurrence(arma_merged, find_anomaly_blocks(arma_merged))
    attacks_ngrams, _ = check_first_occurrence(ngrams_merged, find_anomaly_blocks(ngrams_merged))
    attacks_pca, _ = check_first_occurrence(pca_results, find_anomaly_blocks(pca_results))

    print('------------------------ All sensors considered ------------------------')
    print('True positive: ARMA -> %d N-grams -> %d PCA -> %d' % (tp_arma, tp_ngrams, tp_pca))
    print('False positive: ARMA -> %d N-grams -> %d PCA -> %d' % (fp_arma, fp_ngrams, fp_pca))
    print('True negative: ARMA -> %d N-grams -> %d PCA -> %d' % (tn_arma, tn_ngrams, tn_pca))
    print('False negative: ARMA -> %d N-grams -> %d PCA -> %d' % (fn_arma, fn_ngrams, fn_pca))
    print('Attacks detected: ARMA -> %d N-grams -> %d PCA -> %d' % (attacks_arma, attacks_ngrams, attacks_pca))
    print('Sttd: ARMA -> %d N-grams -> %d PCA -> %d' % (Sttd_arma, Sttd_ngrams, Sttd_pca))
    print('Scm: ARMA -> %d N-grams -> %d PCA -> %d' % (Scm_arma, Scm_ngrams, Scm_pca))
    print('S: ARMA -> %d N-grams -> %d PCA -> %d' % (S_arma, S_ngrams, S_pca))

    prec_arma, rec_arma = pointwise_precision_recall(arma_merged, list(true_anomalies))
    prec_ngrams, rec_ngrams = pointwise_precision_recall(ngrams_merged, list(true_anomalies))
    prec_pca, rec_pca = pointwise_precision_recall(pca_results, list(true_anomalies))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(pd.DataFrame(prec_arma, index=true_anomalies.index), label='ARMA')
    ax.plot(pd.DataFrame(prec_ngrams, index=true_anomalies.index), label='N-grams')
    ax.plot(pd.DataFrame(prec_pca, index=true_anomalies.index), label='PCA')
    ax.xaxis.set_major_locator(mdates.DayLocator([5, 10, 15, 20, 25, 30]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
    plt.xlabel("time")
    plt.ylabel('Precision')
    plt.xticks(rotation=45)
    plt.grid()
    plt.legend()
    plt.savefig('plots/comparison/precision.png', bbox_inches='tight')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(pd.DataFrame(rec_arma, index=true_anomalies.index), label='ARMA')
    ax.plot(pd.DataFrame(rec_ngrams, index=true_anomalies.index), label='N-grams')
    ax.plot(pd.DataFrame(rec_pca, index=true_anomalies.index), label='PCA')
    ax.xaxis.set_major_locator(mdates.DayLocator([5, 10, 15, 20, 25, 30]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
    plt.xlabel("time")
    plt.ylabel('Recall')
    plt.xticks(rotation=45)
    plt.grid()
    plt.legend()
    plt.savefig('plots/comparison/recall.png', bbox_inches='tight')
