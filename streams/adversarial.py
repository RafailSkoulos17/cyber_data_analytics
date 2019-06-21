import pandas as pd
from random import randint, random, sample
import warnings
warnings.filterwarnings("ignore")

# The usage of these perturbation steps and types are inspired from Apruzzese, Giovanni, and Michele Colajanni.
# "Evading Botnet Detectors Based on Flows and Random Forest with Adversarial Samples." 2018 IEEE 17th International
# Symposium on Network Computing and Applications (NCA). IEEE, 2018.
perturbation_steps = {
    1: {'packets': 1, 'bytes': 1},
    2: {'packets': 2, 'bytes': 2},
    3: {'packets': 5, 'bytes': 8},
    4: {'packets': 10, 'bytes': 16},
    5: {'packets': 15, 'bytes': 64},
    6: {'packets': 20, 'bytes': 128},
    7: {'packets': 30, 'bytes': 256},
    8: {'packets': 50, 'bytes': 512},
    9: {'packets': 100, 'bytes': 1024}
}

perturbation_types = {
    1: ['packets'],
    2: ['bytes'],
    3: ['packets', 'bytes'],
}

# infected ips set used for the first version of adversarial dataset creation process
infected_ips = ['147.32.84.155', '147.32.84.156', '147.32.84.157', '147.32.84.158', '147.32.84.159']

# destination ips used for the first version of adversarial dataset creation process consisting of the infected and
# normal hosts in the initial CTU scenario
dest_ips = ['147.32.84.165', '147.32.84.191', '147.32.84.192', '147.32.84.193', '147.32.84.204',
            '147.32.84.205', '147.32.84.206', '147.32.84.207', '147.32.84.208', '147.32.84.209',
            '147.32.84.170', '147.32.84.134', '147.32.84.164', '147.32.87.36', '147.32.80.9', '147.32.87.11']


def make_adversarial_v1(df, altered):
    """
    Function that takes as input the initial malware dataset and adds some handcrafted flows with altered packets or
    (and) bytes. The number of flows added are equal to 1% of the number of Botnet flows already existing. The
    handcrafted flows have a new date (not existing in the initial dataset), a new source and destination ip addresses
    while the rest features (apart from bytes and packets of course) are chosen randomly equal to some other flow in the
    original dataset
    :param df: the dataframe of the initial dataset
    :param altered: the features to be altered
    :return: the new adversarial dataset
    """
    new_df = df.copy()

    # first specify the number of adversarial flows to be constructed - currently 1% of the botnet data
    n = int(0.01*len(df[df['label'] == 'Botnet']))

    # generate n unique randoms indices from the range of the original indices
    inds = sample(range(0, new_df.shape[0]-2), n)

    # create the new unique dates for the new flows
    new_dates = [new_df.index[ind] + random()*(new_df.index[ind+1]-new_df.index[ind]) for ind in inds]
    new_flows = []
    for ind, new_date in zip(inds, new_dates):
        # create new flow
        new_flow = [new_date]
        new_flow += [new_df['duration'][ind]]  # duration
        new_flow += ['TCP']  # and protocol is set to TCP
        new_flow += [infected_ips[randint(0, 4)]]  # choose one of the new ips
        new_flow += [new_df['src_port'][ind]]  # source port
        new_flow += [dest_ips[randint(0, 15)]]  # choose one of the original botnet or normal ips
        new_flow += [new_df['dst_port'][ind]]  # destination port
        new_flow += [new_df['flags'][ind]]  # flags
        new_flow += [new_df['tos'][ind]]  # and tos remain the same
        new_flow += [new_df['packets'][ind] + (0 if 'packets' not in altered.keys() else altered['packets'])]  # alter packets
        new_flow += [new_df['bytes'][ind] + (0 if 'bytes' not in altered.keys() else altered['bytes'])]  # alter bytes
        new_flow += [new_df['flows'][ind]]  # flow remains the same
        new_flow += ['Artificial']  # add the 'Artificial' label just for later plotting issues
        new_flows += [new_flow]

    # create the adversarial dataframe
    column_names = ['date']+list(new_df.columns.values)
    adv_df = pd.DataFrame(new_flows, columns=column_names)
    adv_df = adv_df.set_index(adv_df.date)

    # and concatenate it with the original dataset
    fin_df = pd.concat([new_df, adv_df])
    return fin_df


def make_adversarial_v2(df, altered):
    """
    The second version of adversarial dataset creation. In this version there are not new flows introduced. The botnet
    flows are altered by adding packets or (and) bytes according to the perturbation types and steps specified as input
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

    # create all the adversarial datasets
    for types in perturbation_types.keys():
        for step in perturbation_steps.keys():
            print('Creating perturbation type %d with step %d...' % (types, step))
            to_be_altered = {}
            for type in perturbation_types[types]:
                to_be_altered[type] = perturbation_steps[step][type]
            # adv_df = make_adversarial_v1(data, to_be_altered)  # uncomment this line if you want to run the first
                                                                 # version of adversarial dataset creation
            adv_df = make_adversarial_v2(data, to_be_altered)
            adv_df.to_pickle('adversarial_examples/altered_%s_step_%d.pkl' % ('_'.join(perturbation_types[types]), step))
