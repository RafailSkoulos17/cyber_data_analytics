import pandas as pd
from streams.classification import create_BClus, check_infected
from streams.discretization import find_percentile, netflow_encoding
from streams.profiling import fit_and_apply_hmm, classify
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from imblearn.over_sampling import SMOTE

# The usage of these perturbation steps and types are inspired from Apruzzese, Giovanni, and Michele Colajanni.
# "Evading Botnet Detectors Based on Flows and Random Forest with Adversarial Samples." 2018 IEEE 17th International
# Symposium on Network Computing and Applications (NCA). IEEE, 2018.
perturbation_steps = {
    1: {'packets': 1, 'bytes': 1},
    2: {'packets': 10, 'bytes': 16},
    3: {'packets': 15, 'bytes': 64},
    4: {'packets': 30, 'bytes': 256},
    5: {'packets': 100, 'bytes': 1024}
}

perturbation_types = {
    1: ['packets'],
    2: ['bytes'],
    3: ['packets', 'bytes'],
}

# name the infected hosts
infected_ips = ['147.32.84.165', '147.32.84.191', '147.32.84.192', '147.32.84.193', '147.32.84.204',
                '147.32.84.205', '147.32.84.206', '147.32.84.207', '147.32.84.208', '147.32.84.209']


def make_adversarial(df, altered):
    """
    Function that creates an adversarial dataset in the following way. The botnet flows are altered by adding packets
    or (and) bytes according to the perturbation types and steps specified as input
    :param df: the dataframe of the initial dataset
    :param altered: the features to be altered
    :return: the new adversarial dataset
    """
    new_df = df.copy()
    botnets = new_df[new_df['label'] == 'Botnet']  # keep the botnet flows
    new_df = new_df[new_df['label'] != 'Botnet']  # and remove them from the original dataset

    # alter the packets or (and) the bytes according ot the values of the altered dictionary
    botnets['packets'] = botnets['packets'].apply(lambda z: z + (0 if 'packets' not in altered.keys() else altered['packets']))
    botnets['bytes'] = botnets['bytes'].apply(lambda z: z + (0 if 'bytes' not in altered.keys() else altered['bytes']))

    # and concatenate the new botnet flows with the original dataset with the original dataset
    fin_df = pd.concat([new_df, botnets])
    return fin_df


def make_clf(x_train, y_train, x_test, y_test, clf, clf_name, level):
    """
    Function mostly implemented for the adversarial task - Trains and tests the classifier clf using the initial dataset
    as training set and the adversarial dataset as test set
    The sampling parameter sets the type of sampling to be used
    :param x_train: the original dataset
    :param y_train: the labels of the instances in the original dataset
    :param x_test: the adversarial test set
    :param y_test: the labels of the instances in the adversarial dataset
    :param clf: the classifier to be used
    :param clf_name: the name of the classifier (for plotting reasons)
    :param level: the evaluation level (for plotting reasons)
    :return: the classification results
    """
    print('----------{} at {} level ----------'.format(clf_name, level))
    totalTP, totalFP, totalFN, totalTN = 0, 0, 0, 0

    # apply SMOTE, train and test the model
    x_train, y_train = SMOTE(sampling_strategy=0.5).fit_resample(x_train, y_train)
    clf.fit(x_train, y_train)
    y_predict = clf.predict(x_test)

    for i in range(len(y_predict)):
        if y_test[i] and y_predict[i]:
            totalTP += 1
        if not y_test[i] and y_predict[i]:
            totalFP += 1
        if y_test[i] and not y_predict[i]:
            totalFN += 1
        if not y_test[i] and not y_predict[i]:
            totalTN += 1

    recall = totalTP / (totalTP + totalFN)
    return recall


if __name__ == '__main__':
    # if the data without the background are there, load them (again data from scenario 10 were used)
    data = pd.read_pickle('no_background_data.pkl')

    # resetting indices for data
    data = data.reset_index(drop=True)

    # parse packets and bytes as integers instead of strings
    data['packets'] = data['packets'].astype(int)
    data['bytes'] = data['bytes'].astype(int)

    # sort data by date just to be sure that flows are in chronological order
    data.sort_values('date', ascending=True, inplace=True)

    # set date as index in the dataframe
    data = data.set_index(data.date)
    data.drop('date', axis=1, inplace=True)

    # Create BClus training dataset - to be used for classification
    bclus_data = create_BClus(data)

    # set the classifiers
    clf_name = 'RandomForestClassifier'
    clf = RandomForestClassifier(n_estimators=50, criterion='gini', class_weight='balanced')

    # create all the adversarial datasets
    results_classification_packet = []
    results_classification_host = []
    results_profiling = []
    for types in perturbation_types.keys():
        step_results_classification_packet = []
        step_results_classification_host = []
        step_results_profiling = []
        for step in perturbation_steps.keys():
            print('Creating perturbation type %d with step %d...' % (types, step))
            to_be_altered = {}
            for type in perturbation_types[types]:
                to_be_altered[type] = perturbation_steps[step][type]
            adv_df = make_adversarial(data, to_be_altered)

            # save them just in case
            adv_df.to_pickle('adversarial_examples/altered_%s_step_%d.pkl' % ('_'.join(perturbation_types[types]), step))

            print('Applying flow classification...')
            # Create BClus test dataset with the adversarial dataset
            adv_df['packets'] = adv_df['packets'].astype(int)
            adv_df['bytes'] = adv_df['bytes'].astype(int)
            bclus_test_data = create_BClus(adv_df)

            # enter the classification phase for each level
            eval_levels = ['packet', 'host']  # the 2 evaluation levels
            for level in eval_levels:
                # prepare the data according to the level
                final_data = bclus_data.copy()
                final_test_data = bclus_test_data.copy()

                if level == 'host':
                    final_data = final_data.groupby('src_ip').sum().reset_index()
                    final_test_data = final_test_data.groupby('src_ip').sum().reset_index()

                # label the processed datasets
                final_data['label'] = final_data['src_ip'].apply(lambda z: check_infected(z, infected_ips))
                final_test_data['label'] = final_test_data['src_ip'].apply(lambda z: check_infected(z, infected_ips))

                # separate the labels from the rest of the dataset
                y = final_data['label'].values
                x = final_data.drop(['src_ip', 'label'], axis=1).values

                y_test = final_test_data['label'].values
                x_test = final_test_data.drop(['src_ip', 'label'], axis=1).values

                # enter the classification phase
                print('Start the classification process')
                usx = np.copy(x)
                usy = np.copy(y)
                usx_test = np.copy(x_test)
                usy_test = np.copy(y_test)
                recall = make_clf(usx, usy, usx_test, usy_test, clf, clf_name, level)

                # store the results
                if level == 'packet':
                    step_results_classification_packet += [recall]
                else:
                    step_results_classification_host += [recall]

            print('Discretizing data for the profiling task...')

            # add the numerical representation of the categorical data
            adv_df['protocol_num'] = pd.Categorical(adv_df['protocol'], categories=adv_df['protocol'].unique()).codes
            adv_df['flags_num'] = pd.Categorical(adv_df['flags'], categories=adv_df['flags'].unique()).codes

            # pick one infected host and the normal ones
            infected_ip = '147.32.84.165'
            normal_ips = ['147.32.84.170', '147.32.84.134', '147.32.84.164', '147.32.87.36', '147.32.80.9',
                          '147.32.87.11']

            # list the continuous types of features in the dataset
            continuous_features = ['duration', 'protocol_num', 'flags_num', 'tos', 'packets', 'bytes', 'flows']

            # and select the features that were selected for the profiling task with same number of bins as before
            selected_features = ['protocol', 'bytes']
            for sel in selected_features:
                if sel in continuous_features:
                    percentile_num = 4
                    # assign the cluster id to each value of the selected numerical feature in the way that it is
                    # described in Pellegrino, Gaetano, et al. "Learning Behavioral Fingerprints From Netflows Using
                    # Timed Automata."
                    percentile_values = list(
                        map(lambda p: np.percentile(adv_df[sel], p), 100 * np.arange(0, 1, 1 / percentile_num)[1:]))
                    adv_df[sel + '_num'] = adv_df[sel].apply(find_percentile, args=(percentile_values,))

            # discretize all flows
            print('Discretizing all hosts...')
            mappings = {}
            for sel_feat in selected_features:
                mappings[sel_feat] = len(adv_df[sel_feat + '_num'].unique())
            adv_df['encoded'] = adv_df.apply(lambda x: netflow_encoding(x, mappings), axis=1)

            # proceed to profiling
            print('Profiling in process...')
            chosen = adv_df[(adv_df['src_ip'] == infected_ip) | (adv_df['dst_ip'] == infected_ip)]
            hosts_log_likelihood, modeled_log_likelihood = fit_and_apply_hmm(normal_ips, infected_ips, chosen, adv_df)
            recall = classify(hosts_log_likelihood, normal_ips, infected_ips, modeled_log_likelihood)
            step_results_profiling += [recall]

        results_classification_packet += [step_results_classification_packet]
        results_classification_host += [step_results_classification_host]
        results_profiling += [step_results_profiling]

    # store the final results into dataframe for better visualization and print them
    headers = ['types', 'step 1', 'step 2', 'step 3', 'step 4', 'step 5']

    results_classification_packet = [[' & '.join(prt)] + row for row, prt in
                                     zip(results_classification_packet, list(perturbation_types.values()))]
    results_classification_packet_df = pd.DataFrame(results_classification_packet, columns=headers)
    results_classification_packet_df.set_index('types', inplace=True)
    print('--------- Flow classification results for packet level ---------')
    print(results_classification_packet_df)

    results_classification_host = [[' & '.join(prt)] + row for row, prt in
                                   zip(results_classification_host, list(perturbation_types.values()))]
    results_classification_host_df = pd.DataFrame(results_classification_host, columns=headers)
    results_classification_host_df.set_index('types', inplace=True)
    print('--------- Flow classification results for host level ---------')
    print(results_classification_host_df)

    results_profiling = [[' & '.join(prt)] + row for row, prt
                         in zip(results_profiling, list(perturbation_types.values()))]
    results_profiling_df = pd.DataFrame(results_profiling, columns=headers)
    results_profiling_df.set_index('types', inplace=True)
    print('--------- Profiling results ---------')
    print(results_profiling_df)
