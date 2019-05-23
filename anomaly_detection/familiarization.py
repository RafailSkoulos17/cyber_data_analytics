import pandas as pd
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt


def scale_and_separate(df, labels=True):
    df.columns = df.columns.str.lstrip()
    y = None
    dates = df['DATETIME']
    if labels:
        y = df['ATT_FLAG']
        df.drop(['DATETIME', 'ATT_FLAG'], axis=1, inplace=True)
    else:
        df.drop(['DATETIME'], axis=1, inplace=True)
    scaled_df = pd.DataFrame(StandardScaler().fit_transform(df.values), columns=df.columns)
    scaled_df.insert(loc=0, column='DATETIME', value=dates)
    return scaled_df, y


def make_lineplot(df, feature, dataset, start_date, end_date):
    mask = (df['DATETIME'] >= start_date) & (df['DATETIME'] <= end_date)
    ax = sns.lineplot(x="DATETIME", y=feature, data=df.loc[mask])
    # handles, _ = ax.get_legend_handles_labels()
    # ax.legend(handles, ["benign", "fraudulent"])
    plt.xticks(rotation=45)
    plt.xlabel("time")
    plt.ylabel(feature)
    plt.grid()
    plt.savefig('plots/%s_lineplot_%s.png' % (dataset, feature))


# Dataset columns -> date of the event (hours in a day) -> DATETIME
# tank levels -> L_T1, L_T2, L_T3, L_T4, L_T5, L_T6, L_T7
# status and flow of pumps and valves -> F_PU1, S_PU1, F_PU2, S_PU2, F_PU3, S_PU3, F_PU4, S_PU4, F_PU5, S_PU5, F_PU6,
# S_PU6, F_PU7, S_PU7, F_PU8, S_PU8, F_PU9, S_PU9, F_PU10, S_PU10, F_PU11, S_PU11, F_V2, S_V2
# suction and discharge pressure P_J280, P_J269, P_J300, P_J256, P_J289, P_J415, P_J302, P_J306, P_J307, P_J317, P_J14,
# P_J422
# labels -> ATT_FLAG (normal: 0, under attack: 1, unlabeled: -999)

# train_df1 = pd.read_csv('BATADAL_datasets/BATADAL_training_dataset1.csv', parse_dates=[0],
#                         date_parser=lambda x: pd.to_datetime(x, format="%d/%m/%y %H"))
# scaled_df1, train_y1 = scale_and_separate(train_df1)
# make_lineplot(scaled_df1, 'L_T1')
print('started')
train_df2 = pd.read_csv('BATADAL_datasets/BATADAL_training_dataset2.csv', parse_dates=[0],
                        date_parser=lambda x: pd.to_datetime(x, format="%d/%m/%y %H"))
train_df2.columns = train_df2.columns.str.lstrip()
# print('before scaling')
# scaled_df2, train_y2 = scale_and_separate(train_df2)
print('before plotting')
make_lineplot(train_df2, 'L_T7', 'training2', '15/08/16 00', '15/10/16 00')
#
# test_df = pd.read_csv('BATADAL_datasets/BATADAL_test_dataset.csv', parse_dates=[0],
#                       date_parser=lambda x: pd.to_datetime(x, format="%d/%m/%y %H"))
# scaled_test_df, _ = scale_and_separate(test_df, labels=False)
