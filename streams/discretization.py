import pandas as pd
from functools import reduce
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def netflow_encoding(flow, df, features):
    """
    The netflow encoding described in Pellegrino, Gaetano, et al. "Learning Behavioral Fingerprints From Netflows Using
    Timed Automata."
    :param flow: the flow to be given a code
    :param df: the dataframe with all flows
    :param features: the features to be used for encoding
    :return: the code that represents the flow
    """
    code = 0
    space_size = reduce((lambda x, y: x * y), [len(df[feature+'_num'].unique()) for feature in features])
    for feature in features:
        code += flow[feature+'_num']*space_size/len(df[feature+'_num'].unique())
        space_size = space_size/len(df[feature+'_num'].unique())
    return code


def find_percentile(val, percentiles):
    """
    Helper function returning the relative index of placement in the percentiles
    :param val: the value to be indexed
    :param percentiles: the percentile limits
    :return: the index of val in the percentiles
    """
    ind = len(percentiles)
    for i, p in enumerate(percentiles):
        if val <= p:
            ind = i
            break
    return ind


def remove_background(df):
    """
    Helper function removing background flows from a given dataframe
    :param df: the dataframe
    :return: the no-background dataframe
    """
    df = df[df['labels'] != 'Background']
    return df


if __name__ == '__main__':
    # read the data in chunks due to their large size - uncomment this line if you want to read them again
    # dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f')
    # data = pd.concat(remove_background(chunk) for chunk in pd.read_csv('scenario10/capture20110818.pcap.netflow.labeled_v2',
    #                                                                    chunksize=100000, delimiter=',',
    #                                                                    parse_dates=['date'], date_parser=dateparse))

    # and store them in pickle
    # data.to_pickle('no_background_data.pkl')

    # if the data without the background are there, load them
    data = pd.read_pickle('no_background_data.pkl')
    data = data.reset_index(drop=True)

    # replace the NAN values with zero
    data['duration'] = data['duration'].fillna(0)
    data['tos'] = data['tos'].fillna(0)
    data['packet_bytes'] = data['packet_bytes'].fillna(0)
    data['flows'] = data['flows'].fillna(0)

    # add the numerical representation of the categorical data
    data['protocol_num'] = pd.Categorical(data['protocol'], categories=data['protocol'].unique()).codes
    data['flags_num'] = pd.Categorical(data['flags'], categories=data['flags'].unique()).codes

    infected_ip = '147.32.84.165'
    normal_ips = ['147.32.84.170', '147.32.84.134', '147.32.84.164', '147.32.87.36', '147.32.80.9', '147.32.87.11']

    # currently using only source ips for infected and normal discrimination
    infected = data[data['src_ip'] == infected_ip]
    infected = infected.reset_index()

    normal = data[data['src_ip'].isin(normal_ips)]
    normal = normal.reset_index()

    # continuous features in the dataset
    continuous_features = ['duration', 'protocol_num', 'flags_num', 'tos', 'packet_bytes', 'flows']
    categorical_features = ['protocol', 'flags']

    # check the most discriminative features in the dataset
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(infected[continuous_features].describe())
        print(normal[continuous_features].describe())

    # TODO: add the visualization part

    # and select 2 of them
    selected_features = input('Enter the selected: ').split()

    for sel in selected_features:
        if sel in continuous_features:
            # apply the elbow method
            print('--------------------------------- {} ---------------------------------'.format(sel))
            Sum_of_squared_distances = []
            for k in range(1, 11):
                km = KMeans(n_clusters=k)
                km = km.fit(data[sel].values.reshape(-1, 1))
                Sum_of_squared_distances.append(km.inertia_)

            plt.figure()
            plt.plot(range(1, 11), Sum_of_squared_distances, 'bx-')
            plt.xlabel('k')
            plt.ylabel('Sum_of_squared_distances')
            plt.title('Elbow Method For Optimal k')
            plt.grid()
            # plt.show()
            plt.savefig('plots/elbow_discretization_%s.png' % sel, bbox_inches='tight')

            percentile_num = int(input('Enter your preferred number of clusters: '))

            percentile_values = list(map(lambda p: np.percentile(data[sel], p), 100*np.arange(0, 1, 1 / percentile_num)[1:]))
            data[sel+'_num'] = data[sel].apply(find_percentile, args=(percentile_values,))

    # reselect the data
    infected = data[data['src_ip'] == infected_ip]
    normal = data[data['src_ip'].isin(normal_ips)]

    # discretize the flows
    infected['encoded'] = infected.apply(lambda x: netflow_encoding(x, infected, selected_features), axis=1)
    normal['encoded'] = normal.apply(lambda x: netflow_encoding(x, normal, selected_features), axis=1)
    data['encoded'] = data.apply(lambda x: netflow_encoding(x, data, selected_features), axis=1)

    infected.to_pickle('infected_discretized_%s.pkl' % '_'.join(selected_features))
    normal.to_pickle('normal_discretized_%s.pkl' % '_'.join(selected_features))
    data.to_pickle('all_discretized_%s.pkl' % '_'.join(selected_features))
    # plt.figure()
    # plt.plot(all_encoded)
    # plt.xlabel('records')
    # plt.ylabel('Code')
    # plt.legend()
    # plt.grid()
    # # plt.show()
    # plt.savefig('plots/discretized_flow_all_%s.png' % '_'.join(selected_features), bbox_inches='tight')
