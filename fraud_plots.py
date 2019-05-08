import datetime
import time
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import pandas as pd


def string_to_timestamp(date_string):  # convert time string to float value
    time_stamp = time.strptime(date_string, '%Y-%m-%d %H:%M:%S')
    return time.mktime(time_stamp)


def create_initial_dataset():
    src = 'data_for_student_case.csv'
    ah = open(src, 'r')
    x_labeled = []
    data = []
    conversion_dict = {'SEK': 0.09703, 'MXN': 0.04358, 'AUD': 0.63161, 'NZD': 0.58377, 'GBP': 1.13355}
    ah.readline()  # skip first line
    for line_ah in ah:
        if line_ah.strip().split(',')[9] == 'Refused':
            continue
        if 'na' in str(line_ah.strip().split(',')[14]).lower() or 'na' in str(line_ah.strip().split(',')[4].lower()):
            continue
        issuercountry = line_ah.strip().split(',')[2]  # country code
        txvariantcode = line_ah.strip().split(',')[3]  # type of card: visa/master
        issuer_id = float(line_ah.strip().split(',')[4])  # bin card issuer identifier
        amount = float(line_ah.strip().split(',')[5])  # transaction amount in minor units
        currencycode = line_ah.strip().split(',')[6]
        amount = conversion_dict[currencycode] * amount  # currency conversion
        shoppercountry = line_ah.strip().split(',')[7]  # country code
        interaction = line_ah.strip().split(',')[8]  # online transaction or subscription
        label  = 1 if line_ah.strip().split(',')[9] == 'Chargeback' else 0
        verification = line_ah.strip().split(',')[10]  # shopper provide CVC code or not
        cvcresponse = int(line_ah.strip().split(',')[11])  # 0 = Unknown, 1=Match, 2=No Match, 3-6=Not checked
        if cvcresponse > 2:
            cvcresponse = 3
        year_info = datetime.datetime.strptime(line_ah.strip().split(',')[12], '%Y-%m-%d %H:%M:%S').year
        month_info = datetime.datetime.strptime(line_ah.strip().split(',')[12], '%Y-%m-%d %H:%M:%S').month
        day_info = datetime.datetime.strptime(line_ah.strip().split(',')[12], '%Y-%m-%d %H:%M:%S').day
        creationdate = str(year_info) + '-' + str(month_info) + '-' + str(day_info)  # Date of transaction
        creationdate_stamp = string_to_timestamp(line_ah.strip().split(',')[12])  # Date of transaction-time stamp
        accountcode = line_ah.strip().split(',')[13]  # merchant’s webshop
        mail_id = int(float(line_ah.strip().split(',')[14].replace('email', '')))  # mail
        ip_id = int(float(line_ah.strip().split(',')[15].replace('ip', '')))  # ip
        card_id = int(float(line_ah.strip().split(',')[16].replace('card', '')))  # card
        data.append([issuercountry, txvariantcode, issuer_id, amount, currencycode,
                     shoppercountry, interaction, verification, cvcresponse, creationdate_stamp,
                     accountcode, mail_id, ip_id, card_id, label, creationdate])  # add the interested features here
    data = sorted(data, key=lambda k: k[-1])
    for item in data:  # split data into x,y
        x_labeled.append(item[0:-1])
    df = pd.DataFrame(x_labeled)
    df.columns = ['issuercountry', 'txvariantcode', 'issuer_id', 'amount', 'currencycode',
                  'shoppercountry', 'interaction', 'verification', 'cvcresponse', 'creationdate_stamp',
                  'accountcode', 'mail_id', 'ip_id', 'card_id', 'labels']
    df.to_csv('data_for_plots.csv')


def make_boxplot(data):
    ax = sns.boxplot(x="accountcode", y="amount", hue="labels", data=data,
                palette={0: mcolors.TABLEAU_COLORS['tab:blue'], 1: mcolors.TABLEAU_COLORS['tab:red']}, sym="")
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, ["benign", "fraudulent"])
    plt.xlabel("Merchant's webshop")
    plt.ylabel("Amount in euros")
    plt.grid()
    plt.savefig('plots/boxplot_accountcode_amount.png')
    print('boxplot created')


def make_boxplot_money(data):
    ax = sns.boxplot(x="labels", y="amount", data=data,
                palette={0: mcolors.TABLEAU_COLORS['tab:blue'], 1: mcolors.TABLEAU_COLORS['tab:red']}, sym="")
    ax.set_xticklabels(['benign', 'fraudulent'])
    plt.ylabel("Amount in euros")
    plt.grid()
    plt.savefig('plots/boxplot_labels_amount.png')
    print('boxplot created')


def make_barplot(data):
    cvc_counts = (data.groupby(['labels'])['cvcresponse'].value_counts(normalize=True).rename('percentage').mul(100)
                         .reset_index().sort_values('cvcresponse'))
    ax = sns.barplot(x='cvcresponse', y='percentage', data=cvc_counts, hue='labels',
                     palette={0: mcolors.TABLEAU_COLORS['tab:blue'], 1: mcolors.TABLEAU_COLORS['tab:red']})
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, ["benign", "fraudulent"], loc='upper right')
    plt.xlabel("CVC code")
    plt.ylabel("Percentage of occurrences")
    plt.grid()
    plt.savefig('plots/barplot_cvc.png')
    print('barplot created')


def make_barplot_issued(data):
    cvc_counts = (data.groupby(['labels'])['issuercountry'].value_counts(normalize=True).rename('percentage').mul(100)
                         .reset_index())
    ax = sns.barplot(x='issuercountry', y='percentage', data=cvc_counts, hue='labels',
                     palette={0: mcolors.TABLEAU_COLORS['tab:blue'], 1: mcolors.TABLEAU_COLORS['tab:red']})
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, ["benign", "fraudulent"], loc='upper right')
    ax.set(xlim=(-0.5, 15.5))
    plt.xlabel("Issuer Country")
    plt.ylabel("Percentage of occurrences")
    plt.grid()
    plt.savefig('plots/barplot_issuer.png')
    print('barplot created')


def make_boxplot_card_type(data):
    ax = sns.boxplot(x="amount", y="txvariantcode", hue="labels", data=data,
                     palette={0: mcolors.TABLEAU_COLORS['tab:blue'], 1: mcolors.TABLEAU_COLORS['tab:red']}, sym="")
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, ["benign", "fraudulent"])
    plt.xlabel("Amount in euros")
    plt.ylabel("Type of card")
    plt.grid()
    plt.savefig('plots/boxplot_card_type.png')
    print('boxplot created')


def make_boxplot_issuer_id(data):
    ax = sns.boxplot(x="labels", y="issuer_id", data=data,
                     palette={0: mcolors.TABLEAU_COLORS['tab:blue'], 1: mcolors.TABLEAU_COLORS['tab:red']}, sym="")
    ax.set_xticklabels(['benign', 'fraudulent'])
    plt.ylabel("Card issuer identifier")
    plt.grid()
    plt.savefig('plots/boxplot_labels_issuer_id.png')
    print('boxplot created')


def make_boxplot_ip(data):
    ax = sns.boxplot(x="labels", y="ip_id", data=data,
                     palette={0: mcolors.TABLEAU_COLORS['tab:blue'], 1: mcolors.TABLEAU_COLORS['tab:red']}, sym="")
    ax.set_xticklabels(['benign', 'fraudulent'])
    plt.ylabel("IP address")
    plt.grid()
    plt.savefig('plots/boxplot_labels_ip.png')
    print('boxplot created')


if __name__ == "__main__":
    # create_initial_dataset()
    data = pd.read_csv('data_for_plots.csv')
    plt.figure()
    make_boxplot(data)
    plt.figure()
    make_barplot(data)
    plt.figure()
    make_boxplot_money(data)
    plt.figure()
    make_barplot_issued(data)
    plt.figure(figsize=(13, 8))
    make_boxplot_card_type(data)
    # plt.figure()
    # make_boxplot_issuer_id(data)
    plt.figure()
    make_boxplot_ip(data)
