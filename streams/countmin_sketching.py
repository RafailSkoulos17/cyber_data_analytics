import numpy as np
import pandas as pd
import mmh3
import math
from streams.utils import read_data
import operator
import time


# Class which contains all the functionallity of the COUNT-MIN sketch
class CountMinSketch(object):
    def __init__(self, w, d):
        self.d = d
        self.w = w
        self.cm_array = np.zeros((d, w), dtype=np.int32)

    # Add a new flow to the array
    def add(self, key):
        for i in range(self.d):
            index = mmh3.hash(key, i) % self.w
            self.cm_array[i][index] += 1

    # Find the value of a flow
    def point(self, key):
        min_value = math.inf
        for i in range(self.d):
            index = mmh3.hash(key, i) % self.w
            if self.cm_array[i][index] < min_value:
                min_value = self.cm_array[i][index]
        return min_value


def compute_mse(normal_top, sampled):
    sampled = sampled[:10]
    diff = []
    for index, row in normal_top.iterrows():
        if row['IP'] in sampled['IP'].values:
            normal_freq = row['frequency']
            sampled_freq = sampled.loc[sampled['IP'] == row['IP']].iloc[0]['frequency']
            diff += [abs(normal_freq - sampled_freq) ** 2]
        else:
            diff += [row['frequency'] ** 2]
    mse = math.sqrt(sum(diff))
    return mse


def compute_most_frequent(infected_dataset):
    connections = {}

    # compute the number of flows for each ip
    for index, row in infected_dataset.iterrows():
        src = row['src_ip']
        dst = row['dst_ip']
        if src == infected_host:
            if dst in connections:
                connections[dst] += 1
            else:
                connections[dst] = 1
        elif dst == infected_host:
            if src in connections:
                connections[src] += 1
            else:
                connections[src] = 1
    # sor the results
    sorted_connections = sorted(connections.items(), key=operator.itemgetter(1), reverse=True)
    total_connections = len(infected_dataset)

    # create a dataframe with the frequency ans the number of connections for each ip
    connection_df = pd.DataFrame(sorted_connections, columns=['IP', 'num_of_connections'])
    connection_df['frequency'] = round(100 * connection_df['num_of_connections'] / total_connections, 2)
    return connection_df


def run_cm_sketch(infected_dataset):
    # compute the frequnt ips for different values of d and w
    # we initially define epsilon and delta values we want, and we use these values to compute w and d
    for epsilon in [0.001, 0.005, 0.01, 0.1]:
        for delta in [0.0001, 0.001, 0.005, 0.01, 0.1]:

            # calculate the w, d
            w = int(round(math.e / epsilon))
            d = int(round(math.log(1 / delta)))
            space = w * d
            print('---------- epsilon={}, delta={} ----------'.format(epsilon, delta))
            print('---------- w={}, d={}, space={} ----------\n'.format(w, d, space))

            # construct the matrix with the correct dimensions
            count_min_matrix = CountMinSketch(w, d)

            # add each ip to the matrix
            ips = []
            for index, row in infected_dataset.iterrows():
                src = row['src_ip']
                dst = row['dst_ip']
                if src == infected_host:
                    ips += [dst]
                elif dst == infected_host:
                    ips += [src]

            for ip in ips:
                count_min_matrix.add(ip)

            # find frequency and store it
            count_min = {}
            for ip in set(ips):
                count_min[ip] = count_min_matrix.point(ip)

            total_connections = infected_dataset.shape[0]

            # sort them according to their value to find the 10 most frequent ones
            sorted_count_min = sorted(count_min.items(), key=operator.itemgetter(1), reverse=True)
            min_sketch = pd.DataFrame(sorted_count_min, columns=['IP', 'num_of_connections'])
            min_sketch['frequency'] = round(100 * min_sketch['num_of_connections'] / total_connections, 2)
            print(min_sketch[:10], '\n')
            min_sketch_ips = min_sketch[:10]['IP'].tolist()
            print('\nDifferent IPs: {}'.format(len(set(normal_top_ips) - set(min_sketch_ips))))
            mse = compute_mse(normal_top, min_sketch)
            print("Frequency difference: %0.3f\n" % mse)


def run_cm_sketch_comparison(infected_dataset):
    # we initially define w and d
    w_d_list = [(50, 2), (25, 4), (250, 4), (500, 2), (125, 8), (1250, 4), (2500, 2), (625, 8)]

    for l in w_d_list:
        w, d = l
        space = w * d
        print('---------- w={}, d={}, space={} ----------\n'.format(w, d, space))

        start = time.time()

        # construct the matrix with the correct dimensions
        count_min_matrix = CountMinSketch(w, d)

        # add each ip to the matrix
        ips = []
        for index, row in infected_dataset.iterrows():
            src = row['src_ip']
            dst = row['dst_ip']
            if src == infected_host:
                ips += [dst]
            elif dst == infected_host:
                ips += [src]

        for ip in ips:
            count_min_matrix.add(ip)

        # find frequency and store it
        count_min = {}
        for ip in set(ips):
            count_min[ip] = count_min_matrix.point(ip)

        total_connections = infected_dataset.shape[0]
        # sort them according to their value to find the 10 most frequent ones
        sorted_count_min = sorted(count_min.items(), key=operator.itemgetter(1), reverse=True)

        # stop time recording
        stop = time.time()

        min_sketch = pd.DataFrame(sorted_count_min, columns=['IP', 'num_of_connections'])
        min_sketch['frequency'] = round(100 * min_sketch['num_of_connections'] / total_connections, 2)

        print(min_sketch[:10])
        min_sketch_ips = min_sketch[:10]['IP'].tolist()
        print('\nExecution time: ', stop - start)
        print('Different IPs: {}'.format(len(set(normal_top_ips) - set(min_sketch_ips))))
        mse = compute_mse(normal_top, min_sketch)
        print("Frequency difference: %0.3f\n" % mse)


if __name__ == '__main__':
    infected_host = '147.32.84.165'

    data = read_data('datasets/CTU-Malware-Capture-Botnet-54')
    # data.to_pickle('./data.pkl')
    # data = pd.read_pickle('./data.pkl')
    infected_dataset = data.loc[(data['src_ip'] == infected_host) | (data['dst_ip'] == infected_host)]
    print('Rows with infected host: {}'.format(infected_dataset.shape[0]))

    # compute the true top 10 IPs
    normal_top = compute_most_frequent(infected_dataset)[:10]
    normal_top_ips = normal_top['IP'].tolist()

    # Perform the COUNT-MIN sketch and find the top 10 IPs for different values of w and d
    run_cm_sketch(infected_dataset)

    # Perform the COUNT-MIN sketch for standard w and d values,
    # and measure execution time and compare with Reservoir Sampling
    run_cm_sketch_comparison(infected_dataset)
