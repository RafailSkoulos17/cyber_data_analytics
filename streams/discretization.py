import pandas as pd
from functools import reduce
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def make_barplot(data, feature):
    """
    Function for visualising the difference between categorical features for infected and normal hosts
    :param data: the dataframe containing the data
    :return: creates the wanted plot
    """
    plt.figure()
    feature_counts = (data.groupby(['is_infected'])[feature].value_counts(normalize=True).rename('percentage').mul(100)
                         .reset_index().sort_values(feature))
    ax = sns.barplot(x=feature, y='percentage', data=feature_counts, hue='is_infected',
                     palette={0: mcolors.TABLEAU_COLORS['tab:blue'], 1: mcolors.TABLEAU_COLORS['tab:red']})
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, ['Normal Hosts', 'Infected host'], loc='upper right')
    plt.xlabel("%s type" % feature)
    plt.ylabel("Percentage of occurrences")
    plt.grid()
    plt.savefig('plots/barplot_%s.png' % feature)


def make_bar_graphs(x, y, feature):
    plt.figure()
    y_pos = np.arange(len(x))
    plt.bar(y_pos, y, align='center', alpha=0.5, color=[mcolors.TABLEAU_COLORS['tab:blue'],
                                                        mcolors.TABLEAU_COLORS['tab:red']])
    plt.xticks(y_pos, x)
    plt.xlabel('Type of host')
    plt.ylabel(feature)
    plt.title('Average number of %s sent' % feature)

    plt.savefig('plots/barplot_%s.png' % feature)


def netflow_encoding(flow, mappings):
    """
    The netflow encoding described in Pellegrino, Gaetano, et al. "Learning Behavioral Fingerprints From Netflows Using
    Timed Automata."
    :param flow: the flow to be given a code
    :param df: the dataframe with all flows
    :param mappings: dictionary with the features to be used for encoding and their cardinality
    :return: the code that represents the flow
    """
    code = 0
    space_size = reduce((lambda x, y: x * y), list(mappings.values()))
    for feature in mappings.keys():
        code += flow[feature+'_num']*space_size/mappings[feature]
        space_size = space_size/mappings[feature]
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
    df = df[df['label'] != 'Background']
    return df


if __name__ == '__main__':
    # # read the data in chunks due to their large size - uncomment this line if you want to read them again
    # dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f')
    # data = pd.concat(remove_background(chunk) for chunk in pd.read_csv('scenario10/capture20110818.pcap.netflow.labeled_v2',
    #                                                                    chunksize=100000, delimiter=',',
    #                                                                    parse_dates=['date'], date_parser=dateparse))

    # # and store them in pickle
    # data.to_pickle('no_background_data.pkl')

    # if the data without the background are there, load them
    data = pd.read_pickle('no_background_data.pkl')

    # resetting indices for data
    data = data.reset_index(drop=True)

    # replace the NAN values with zero
    data['duration'] = data['duration'].fillna(0)
    data['packets'] = data['packets'].fillna(0)
    data['bytes'] = data['bytes'].fillna(0)

    # parse packets and bytes as integers instead of strings
    data['packets'] = data['packets'].astype(int)
    data['bytes'] = data['bytes'].astype(int)

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
    continuous_features = ['duration', 'protocol_num', 'flags_num', 'tos', 'packets', 'bytes', 'flows']
    categorical_features = ['protocol', 'flags']

    # check the most discriminative features in the dataset
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print('---------------- Stats for infected host ----------------')
        print(infected[continuous_features].describe())
        print('---------------- Stats for normal hosts ----------------')
        print(normal[continuous_features].describe())

    # preprocess the infected and normal data and plot possible feature candidates
    plot_data = pd.concat([infected, normal], ignore_index=True)
    for index, ip in enumerate(plot_data['src_ip']):
        val = 1 if ip == infected_ip else 0
        plot_data.set_value(index, 'is_infected', val)

    make_barplot(plot_data, 'protocol')
    make_bar_graphs(['Normal Hosts', 'Infected Host'], [normal["packets"].mean(), infected["packets"].mean()], 'packets')
    make_bar_graphs(['Normal Hosts', 'Infected Host'], [normal["bytes"].mean(), infected["bytes"].mean()], 'bytes')
    print('plots created')

    # and select 2 of them
    selected_features = input('Enter the selected: ').split()

    for sel in selected_features:
        if sel in continuous_features:
            # apply the elbow method
            print('----------------------- Finding optimal number of bins for {} -----------------------'.format(sel))
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
    infected = infected.reset_index()
    normal = data[data['src_ip'].isin(normal_ips)]
    normal = normal.reset_index()

    # discretize the infected flows
    print('Discretizing the infected host...')
    mappings = {}
    for sel_feat in selected_features:
        mappings[sel_feat] = len(infected[sel_feat+'_num'].unique())

    infected['encoded'] = infected.apply(lambda x: netflow_encoding(x, mappings), axis=1)
    infected.to_pickle('infected_discretized_%s.pkl' % '_'.join(selected_features))

    print('Discretizing the normal hosts...')
    mappings = {}
    for sel_feat in selected_features:
        mappings[sel_feat] = len(normal[sel_feat + '_num'].unique())
    normal['encoded'] = normal.apply(lambda x: netflow_encoding(x, mappings), axis=1)
    normal.to_pickle('normal_discretized_%s.pkl' % '_'.join(selected_features))

    print('Discretizing all hosts...')
    mappings = {}
    for sel_feat in selected_features:
        mappings[sel_feat] = len(data[sel_feat + '_num'].unique())
    data['encoded'] = data.apply(lambda x: netflow_encoding(x, mappings), axis=1)
    data.to_pickle('all_discretized_%s.pkl' % '_'.join(selected_features))
