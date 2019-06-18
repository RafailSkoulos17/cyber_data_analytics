import pandas as pd
from random import randint, random
import warnings
warnings.filterwarnings("ignore")


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


infected_ips = ['147.32.84.165', '147.32.84.191', '147.32.84.192', '147.32.84.193', '147.32.84.204',
                '147.32.84.205', '147.32.84.206', '147.32.84.207', '147.32.84.208', '147.32.84.209']


def insert_row(row_number, df, row_value):
    start_upper = 0
    end_upper = row_number
    start_lower = row_number
    end_lower = df.shape[0]

    # Create a list of upper_half index
    upper_half = [*range(start_upper, end_upper, 1)]

    # Create a list of lower_half index
    lower_half = [*range(start_lower, end_lower, 1)]

    # Increment the value of lower half by 1
    lower_half = [x.__add__(1) for x in lower_half]

    # Combine the two lists
    index_ = upper_half + lower_half

    # Update the index of the dataframe
    df.index = index_

    # Insert a row at the end and sort the index labels
    df.loc[row_number] = row_value
    df = df.sort_index()

    return df


def make_adversarial(df, altered):
    new_df = df.copy()
    new_df.sort_values('date', ascending=True, inplace=True)
    # first specify the number of adversarial flows to be constructed
    n = int(0.25*len(df[df['label'] == 'Botnet']))
    for i in range(n):
        ind = randint(0, new_df.shape[0]-2)
        new_flow = []
        new_flow += [new_df['date'][ind] + random()*(new_df['date'][ind+1]-new_df['date'][ind])]
        new_flow += [new_df['duration'][ind]]
        new_flow += [new_df['protocol'][ind]]
        new_flow += [infected_ips[randint(0, 9)]]
        new_flow += [new_df['src_port'][ind]]
        new_flow += [new_df['dst_ip'][ind]]
        new_flow += [new_df['dst_port'][ind]]
        new_flow += [new_df['flags'][ind]]
        new_flow += [new_df['tos'][ind]]
        new_flow += [new_df['packets'][ind] + 0 if 'packets' not in altered.keys() else altered['packets']]
        new_flow += [new_df['bytes'][ind] + 0 if 'bytes' not in altered.keys() else altered['bytes']]
        new_flow += [new_df['flows'][ind]]
        new_flow += ['Artificial']
        new_df = insert_row(ind+1, new_df, new_flow)
    return new_df


if __name__ == '__main__':
    # if the data without the background are there, load them (again data from scenario 10 were used)
    data = pd.read_pickle('no_background_data.pkl')

    # resetting indices for data
    data = data.reset_index(drop=True)

    # parse packets and bytes as integers instead of strings
    data['packets'] = data['packets'].astype(int)
    data['bytes'] = data['bytes'].astype(int)

    for types in perturbation_types.keys():
        for step in perturbation_steps.keys():
            to_be_altered = {}
            for type in perturbation_types[types]:
                to_be_altered[type] = perturbation_steps[step][type]
            adv_df = make_adversarial(data, to_be_altered)
            adv_df.to_pickle('altered_%s_step_%d.pkl' % ('_'.join(perturbation_types[types]), step))

