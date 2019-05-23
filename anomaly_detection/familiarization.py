import pandas as pd


train_df1 = pd.read_csv('BATADAL_datasets/BATADAL_training_dataset1.csv', parse_dates=[0],
                        date_parser=lambda x: pd.to_datetime(x, format="%d/%m/%y %H"))
train_df2 = pd.read_csv('BATADAL_datasets/BATADAL_training_dataset2.csv', parse_dates=[0],
                        date_parser=lambda x: pd.to_datetime(x, format="%d/%m/%y %H"))
test_df = pd.read_csv('BATADAL_datasets/BATADAL_test_dataset.csv', parse_dates=[0],
                      date_parser=lambda x: pd.to_datetime(x, format="%d/%m/%y %H"))
