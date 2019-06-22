import math
import operator
import random
import time

from streams.utils import read_data
import pandas as pd

infected_host = '147.32.84.165'
# data = read_data()
# data.to_pickle('./data.pkl')
data = pd.read_pickle('./data.pkl')
infected_dataset = data.loc[(data['src'] == infected_host) | (data['dst'] == infected_host)]


# Finds the frequency and the number of flows for each IP in the infected dataset
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


# Performs tha Reservoir Sampling
def reservoir_sampling(infected_dataset, k):
    result = []
    for i, (index, row) in enumerate(infected_dataset.iterrows()):
        i += 1
        if len(result) < k:
            result.append(index)
        else:
            s = random.randint(1, i)
            if s < k:
                result[s] = index
    return data.iloc[result]


# Finds the difference in the frequence for the top 10 IPs , between the true sequence
# and the one obtained from Reservoir sampling
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


def run_reservoir(normal_top):
    k_values = [100, 500, 1000, 5000, 10000, 20000]
    for k in k_values:
        start = time.time()
        res = reservoir_sampling(infected_dataset, k)
        reservoir_top = compute_most_frequent(res)[:10]
        stop = time.time()
        reservoir_top_ips = reservoir_top['IP'].tolist()
        normal_top_ips = normal_top['IP'].tolist()
        print('---------- k = {} ----------\n'.format(k))
        print(reservoir_top)
        print('\nDifferent IPs: {}'.format(len(set(normal_top_ips) - set(reservoir_top_ips))))
        mse = compute_mse(normal_top, compute_most_frequent(res))
        print("Frequency difference: %0.3f" % mse)
        print('Execution time: ', stop - start, '\n')


if __name__ == '__main__':
    # initialize random seed to get always the same results
    random.seed(0)
    infected_host = '147.32.84.165'

    # uncomment to read the data
    data = read_data('datasets/CTU-Malware-Capture-Botnet-54')
    # data.to_pickle('./data.pkl')

    # load the dataset
    # data = pd.read_pickle('./data.pkl')
    infected_dataset = data.loc[(data['src_ip'] == infected_host) | (data['dst_ip'] == infected_host)]
    print('Flows with infected host: {}'.format(infected_dataset.shape[0]))

    # Find the 10 most frequent IPs of the stream
    normal_top = compute_most_frequent(infected_dataset)[:10]
    print(normal_top)

    # Find the 10 most frequent IPs of the stream by performing Reservoir Sampling
    # for several reservoir values
    run_reservoir(normal_top)
