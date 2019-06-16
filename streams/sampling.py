import operator
import random
from streams.utils import read_data
import pandas as pd

infected_host = '147.32.84.165'
data = read_data()
data.to_pickle('./data.pkl')
# data = pd.read_pickle('./data.pkl')
infected_dataset = data.loc[(data['src'] == infected_host) | (data['dst'] == infected_host)]


def compute_10_most_frequent(infected_dataset):
    connections = {}

    for index, row in infected_dataset.iterrows():
        src = row['src']
        dst = row['dst']
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

    sorted_connections = sorted(connections.items(), key=operator.itemgetter(1), reverse=True)
    total_connections = len(infected_dataset)

    connection_df = pd.DataFrame(sorted_connections, columns=['IP', 'num_of_connections'])
    connection_df['percentage'] = round(100 * connection_df['num_of_connections'] / total_connections, 2)
    return connection_df[:10]


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


normal_top = compute_10_most_frequent(infected_dataset)

res = reservoir_sampling(infected_dataset, 1000)
reservoir_top = compute_10_most_frequent(res)
