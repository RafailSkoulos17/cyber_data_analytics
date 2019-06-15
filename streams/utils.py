import pandas as pd


def read_data():
    preprocess_data()
    dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f')
    data = pd.read_csv('capture20110815-3.pcap.netflow.labeled_v2', delimiter=',',
                       parse_dates=['date'], date_parser=dateparse)
    return data


def preprocess_data():
    with open('capture20110815-3.pcap.netflow.labeled', 'r') as f:
        lines = f.readlines()
    fout = open('capture20110815-3.pcap.netflow.labeled_v2', 'w')
    column_names = ['date', 'durat', 'prot', 'src', 'dst', 'flags',
                    'tos', 'packet_bytes', 'flows', 'label', 'labels']
    fout.write(','.join(column_names))
    fout.write('\n')
    for line in lines[1:]:
        elements = []
        columns = line.split()
        elements += [columns[0] + ' ' + columns[1]]
        elements += [columns[2]]
        elements += [columns[3]]
        elements += [columns[4]]
        elements += [columns[6]]
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
    data = read_data()
