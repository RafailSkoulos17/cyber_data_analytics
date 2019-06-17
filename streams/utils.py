import pandas as pd


def read_data(filepath):
    preprocess_data(filepath)
    dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f')
    data = pd.read_csv(filepath+'_v2', delimiter=',', parse_dates=['date'], date_parser=dateparse)
    return data


def preprocess_data(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    fout = open(filepath+'_v2', 'w')
    column_names = ['date', 'duration', 'protocol', 'src_ip', 'src_port', 'dst_ip', 'dst_port', 'flags',
                    'tos', 'packets', 'bytes', 'flows', 'label']
    fout.write(','.join(column_names))
    fout.write('\n')
    for line in lines[1:]:
        elements = []
        columns = line.split()
        elements += [columns[0] + ' ' + columns[1]]
        elements += [columns[2]]
        elements += [columns[3]]
        elements += [columns[4].split(':')[0]]
        elements += ['na' if len(columns[4].split(':')) == 1 else columns[4].split(':')[1]]
        elements += [columns[6].split(':')[0]]
        elements += ['na' if len(columns[6].split(':')) == 1 else columns[6].split(':')[1]]
        elements += [columns[7]]
        elements += [columns[8]]
        elements += [columns[9]]
        elements += [columns[10]]
        elements += [columns[11]]
        elements += [columns[12]]
        fout.write(','.join(elements))
        fout.write('\n')
    fout.close()


if __name__ == '__main__':
    data = preprocess_data('scenario10/capture20110818.pcap.netflow.labeled')
    print('scenario 10 read')
