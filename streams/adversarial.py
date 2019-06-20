import pandas as pd
from random import randint, random, sample
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


def make_adversarial(df, altered):
    new_df = df.copy()
    # first specify the number of adversarial flows to be constructed - currently 1% of the botnet data
    n = int(0.01*len(df[df['label'] == 'Botnet']))
    inds = sample(range(0, new_df.shape[0]-2), n)
    new_dates = [new_df.index[ind] + random()*(new_df.index[ind+1]-new_df.index[ind]) for ind in inds]
    new_flows = []
    for ind, new_date in zip(inds, new_dates):
        # ind = randint(0, new_df.shape[0]-2)

        # create a new date that does not already exist in the dataset
        # new_date = new_df.index[ind] + random()*(new_df.index[ind+1]-new_df.index[ind])

        # create new flow
        new_flow = [new_date]
        new_flow += [new_df['duration'][ind]]  # duration
        new_flow += [new_df['protocol'][ind]]  # and protocol remain the same
        new_flow += [infected_ips[randint(0, 9)]]  # choose one of the botnet ips
        new_flow += [new_df['src_port'][ind]]  # source port
        new_flow += [new_df['dst_ip'][ind]]  # destination ip
        new_flow += [new_df['dst_port'][ind]]  # destination port
        new_flow += [new_df['flags'][ind]]  # flags
        new_flow += [new_df['tos'][ind]]  # and tos remain the same
        new_flow += [new_df['packets'][ind] + 0 if 'packets' not in altered.keys() else altered['packets']]  # alter packets
        new_flow += [new_df['bytes'][ind] + 0 if 'bytes' not in altered.keys() else altered['bytes']]  # alter bytes
        new_flow += [new_df['flows'][ind]]  # flow remains the same
        new_flow += ['Artificial']  # add the 'Artificial' label just for later plotting issues

        # TODO: check how to optimize it for memory
        # new_df.loc[new_date] = new_flow  # add the new flow to the dataset
        new_flows += [new_flow]
    column_names = ['date']+list(new_df.columns.values)
    adv_df = pd.DataFrame(new_flows, columns=column_names)
    adv_df = adv_df.set_index(adv_df.date)
    fin_df = pd.concat([new_df, adv_df])
    return fin_df


if __name__ == '__main__':
    # if the data without the background are there, load them (again data from scenario 10 were used)
    data = pd.read_pickle('no_background_data.pkl')

    # resetting indices for data
    data = data.reset_index(drop=True)

    # parse packets and bytes as integers instead of strings
    data['packets'] = data['packets'].astype(int)
    data['bytes'] = data['bytes'].astype(int)

    data.sort_values('date', ascending=True, inplace=True)

    # set date as index in the dataframe
    data = data.set_index(data.date)
    data.drop('date', axis=1, inplace=True)

    # create all the adversarial datasets
    for types in perturbation_types.keys():
        for step in perturbation_steps.keys():
            print('creating type %d with step %d' % (types, step))
            to_be_altered = {}
            for type in perturbation_types[types]:
                to_be_altered[type] = perturbation_steps[step][type]
            adv_df = make_adversarial(data, to_be_altered)
            adv_df.to_pickle('altered_%s_step_%d.pkl' % ('_'.join(perturbation_types[types]), step))
