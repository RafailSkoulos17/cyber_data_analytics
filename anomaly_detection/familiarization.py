import pandas as pd
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
# from dtaidistance import dtw
# from dtaidistance import dtw_visualisation as dtwvis
import numpy as np


def scale_and_separate(df, labels=True):
    """
    Function that applies normalization to the data by subtracting the mean and dividing by the std for every
    observation in each column
    :param df: the initial dataframe
    :param labels: if there are attack labels in the dataset
    :return: the scaled dataset with the labels separated (if existed)
    """
    df.columns = df.columns.str.lstrip()
    y = None
    if labels:
        y = df['ATT_FLAG']
        df.drop(['ATT_FLAG'], axis=1, inplace=True)
    scaled_df = pd.DataFrame(StandardScaler().fit_transform(df.values), index=df.index, columns=df.columns)
    return scaled_df, y


def make_lineplot(df, features, dataset, start_date, end_date):
    """
    Function that produces the lineplots for the visualization of the signals
    :param df: the dataframe with the signals
    :param features: which signals to plot
    :param dataset: the type of the dataset (training or test set)
    :param start_date: the starting date in the plot
    :param end_date: the ending date in the plot
    :return: creates and saves the produced lineplot
    """
    # axes = df.loc[start_date:end_date][features].plot(figsize=(11, 9), subplots=True)
    fig, ax = plt.subplots(len(features), 1, sharex=True)
    colors = ['b', 'r', 'g']
    for ind, feature in enumerate(features):
        ax[ind].plot(df.loc[start_date:end_date][feature], colors[ind])
        ax[ind].xaxis.set_major_locator(mdates.DayLocator([5, 10, 15, 20, 25, 30]))
        ax[ind].xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
        ax[ind].set_ylabel(feature)
        ax[ind].grid()
    plt.xticks(rotation=45)
    plt.xlabel("time")
    if len(features) > 1:
        plt.savefig('plots/%s/lineplot_%s.png' % (dataset, '_'.join(features)), bbox_inches='tight')
    else:
        plt.savefig('plots/%s/lineplot_%s.png' % (dataset, features[0]), bbox_inches='tight')

# Dataset columns -> date of the event (hours in a day) -> DATETIME
# tank levels -> L_T1, L_T2, L_T3, L_T4, L_T5, L_T6, L_T7
# status and flow of pumps and valves -> F_PU1, S_PU1, F_PU2, S_PU2, F_PU3, S_PU3, F_PU4, S_PU4, F_PU5, S_PU5, F_PU6,
# S_PU6, F_PU7, S_PU7, F_PU8, S_PU8, F_PU9, S_PU9, F_PU10, S_PU10, F_PU11, S_PU11, F_V2, S_V2
# suction and discharge pressure P_J280, P_J269, P_J300, P_J256, P_J289, P_J415, P_J302, P_J306, P_J307, P_J317, P_J14,
# P_J422
# labels -> ATT_FLAG (normal: 0, under attack: 1, unlabeled: -999)

# training dataset 1 with no attacks from 06/01/14 00 to 06/01/15 00
# train_df1 = pd.read_csv('BATADAL_datasets/BATADAL_training_dataset1.csv', parse_dates=[0],
#                         date_parser=lambda x: pd.to_datetime(x, format="%d/%m/%y %H"))
# scaled_df1, train_y1 = scale_and_separate(train_df1)


# training dataset 2 with attack from 04/07/16 00 to 25/12/16 00
print('Reading the dataset')
train_df2 = pd.read_csv('BATADAL_datasets/BATADAL_training_dataset2.csv', index_col=0, parse_dates=[0],
                        date_parser=lambda x: pd.to_datetime(x, format="%d/%m/%y %H"))
# train_df2.columns = train_df2.columns.str.lstrip()
print('Applying scaling on the data')
scaled_df2, train_y2 = scale_and_separate(train_df2)
print('Plotting')
make_lineplot(scaled_df2, ['L_T7', 'F_PU10', 'F_PU11'], 'training2', '2016-09-01', '2016-09-30')
make_lineplot(scaled_df2, ['L_T1', 'F_PU1', 'F_PU2'], 'training2', '2016-10-01', '2016-11-10')
make_lineplot(train_df2, ['L_T1', 'S_PU1', 'S_PU2'], 'training2', '2016-10-01', '2016-11-10')
make_lineplot(scaled_df2, ['L_T4', 'F_PU7'], 'training2', '2016-11-25', '2016-12-20')

# test dataset with attacks from 04/01/17 00 to 01/04/17 00
print('Reading the dataset')
test_df = pd.read_csv('BATADAL_datasets/BATADAL_test_dataset.csv', index_col=0, parse_dates=[0],
                      date_parser=lambda x: pd.to_datetime(x, format="%d/%m/%y %H"))
print('Applying scaling on the data')
scaled_test_df, _ = scale_and_separate(test_df, labels=False)
print('Plotting')
make_lineplot(scaled_test_df, ['L_T7', 'F_PU10', 'F_PU11'], 'test', '2016-09-01', '2016-09-30')
make_lineplot(scaled_test_df, ['L_T1', 'F_PU1', 'F_PU2'], 'test', '2016-10-01', '2016-11-10')
make_lineplot(train_df2, ['L_T1', 'S_PU1', 'S_PU2'], 'test', '2016-10-01', '2016-11-10')
make_lineplot(scaled_test_df, ['L_T4', 'F_PU7'], 'test', '2016-11-25', '2016-12-20')