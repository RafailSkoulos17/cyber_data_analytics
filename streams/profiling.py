import math

from hmmlearn.hmm import GaussianHMM
import pandas as pd
import numpy as np
import pickle


def get_windows(data, window_size):
    size = len(data) - window_size
    # create sliding window data
    win_data = np.zeros((size, window_size), dtype=np.int32)
    for i in range(size):
        win_data[i] = np.array([flow for flow in data['encoded'].iloc[i:i + window_size]])
    return win_data


def find_optimal_params(chosen):
    max_ll = -math.inf
    optimal_win = 0
    optimal_components = 0
    for comp in range(2, 7):
        for win in range(2, 11):
            win_data = get_windows(chosen, win)
            hmm = GaussianHMM(n_components=comp)
            hmm.fit(win_data)
            log_likelihood = hmm.decode(win_data)[0]
            if log_likelihood > max_ll:
                max_ll = log_likelihood
                optimal_win = win
                optimal_components = comp
            # uncomment to see the log-likelihood for each configuration
            # print('Window=%d ,Components=%d, Log-likelihood=%.3f' % (win, comp, log_likelihood))
    print('Profiling: Optimal HMM values: Window=%d, Components=%d' % (optimal_win, optimal_components))
    return optimal_win, optimal_components


def fit_and_apply_hmm(normal, infected, chosen, data):
    # define sliding window size and number of components
    win, components = 4, 5
    # uncomment the next line to find the optimal window size and number of components
    # it takes some time though...
    # win, components = find_optimal_params(chosen)

    win_data = get_windows(chosen, win)

    # learn a Gaussian Hidden Markov Model with 4 states from the infected host data
    hmm = GaussianHMM(n_components=components)
    hmm.fit(win_data)
    # store the log-likelihood of the host that trained the model
    modeled_log_likelihood = hmm.decode(win_data)[0]

    hosts_log_likelihood = {}

    # compute log-likelihood of data sequence of normal IPs
    for ip in normal:
        # get the flows of that host only
        host_data = data[(data['src_ip'] == ip) | (data['dst_ip'] == ip)]
        size = len(host_data) - win
        # if host has enough flows for creating a window
        if size > 0:
            # create sliding windows sequences
            normal_data = get_windows(host_data, win)
            # get the log-likelihood of the sequential data
            hosts_log_likelihood[ip] = hmm.decode(normal_data)[0]
        else:
            hosts_log_likelihood[ip] = 0

    # repeat procedure for all infected IPs
    for ip in infected:
        # get the flows of that host only
        host_data = data[(data['src_ip'] == ip) | (data['dst_ip'] == ip)]
        size = len(host_data) - win
        # if host has enough flows for creating a window
        if size > 0:
            # create sliding windows sequences
            infected_data = get_windows(host_data, win)
            # get the log-likelihood of the sequential data
            hosts_log_likelihood[ip] = hmm.decode(infected_data)[0]
        else:
            hosts_log_likelihood[ip] = 0
    return hosts_log_likelihood, modeled_log_likelihood


def classify(hosts_log_likelihood, normal, infected, modeled_log_likelihood):
    # evaluate results using the log-likelihood distance of hosts from the one who trained the model
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    positives = []
    negatives = []

    dist = {}
    for ip in hosts_log_likelihood.keys():
        # absolute log-likelihood distance
        dist[ip] = abs(hosts_log_likelihood[ip] - modeled_log_likelihood)
        #     print('IP = {}, Diff= {}, abs(ll)/2 = {}'.format(ip, dist[ip],abs(ll) / 2))
        # threshold is half log-likelihood
        if dist[ip] > modeled_log_likelihood / 2:
            negatives.append(ip)
        else:
            positives.append(ip)

    # evaluate all potentially malicious hosts
    for i in positives:
        if i in infected:
            TP += 1
        else:
            FP += 1

    # evaluate all potentially benign hosts
    for i in negatives:
        if i in normal:
            TN += 1
        else:
            FN += 1

    if not (TP + FP):
        precision = 1
    else:
        precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    return recall, precision, accuracy, TP, FP, TN, FN  # for task 6


if __name__ == '__main__':
    with open('discretized_data/all_discretized_protocol_bytes.pkl', 'rb') as f:
        data = pickle.load(f)

    infected_ip = '147.32.84.165'

    # the infected host flows that we will profile
    chosen = data[(data['src_ip'] == infected_ip) | (data['dst_ip'] == infected_ip)]
    # rest of the hosts split between benign and malicious for testing purposes
    normal = ['147.32.84.170', '147.32.84.134', '147.32.84.164', '147.32.87.36', '147.32.80.9', '147.32.87.11']
    infected = ['147.32.84.191', '147.32.84.192', '147.32.84.193', '147.32.84.204', '147.32.84.205',
                '147.32.84.206', '147.32.84.207', '147.32.84.208', '147.32.84.209']

    hosts_log_likelihood, modeled_log_likelihood = fit_and_apply_hmm(normal, infected, chosen, data)
    recall, precision, accuracy, TP, FP, TN, FN = classify(hosts_log_likelihood, normal, infected,
                                                           modeled_log_likelihood)
    print('True Positives : {}'.format(TP))
    print('False Positives : {}'.format(FP))
    print('True Negatives : {}'.format(TN))
    print('False Negatives : {}'.format(FN))
    print('Precision: {}'.format(precision))
    print('Recall: {}'.format(recall))
    print('Accuracy: {}'.format(accuracy))
