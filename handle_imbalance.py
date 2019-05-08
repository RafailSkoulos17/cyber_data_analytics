import time
from sklearn import neighbors
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier, VotingClassifier, \
    GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler


def plot_roc(fpr, tpr, roc_auc, fpr_smote, tpr_smote, roc_auc_smote, clf_name):
    plt.figure()
    plt.title('{} - Receiver Operating Characteristic'.format(clf_name))
    plt.plot(fpr, tpr, 'r', label='AUC unSMOTEd = %0.2f' % roc_auc)
    plt.plot(fpr_smote, tpr_smote, 'g', label='AUC SMOTEd = %0.2f' % roc_auc_smote)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'b--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig('imbalance_plots/{}_ROC.png'.format(clf_name), bbox_inches='tight')


def plot_confusion_matrix(y_true, y_pred, clf_name, classes, smote, normalize=False, title=None, cmap=plt.cm.Blues):
    if not title:
        if normalize:
            title = '{} - Normalized confusion matrix'.format(clf_name+smote)
        else:
            title = '{} - Confusion matrix, without normalization'.format(clf_name+smote)
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig('imbalance_plots/{}_confusion.png'.format(clf_name+smote))
    return ax


def plot_prec_rec(precision, recall, precision_smote, recall_smote):
    plt.figure()
    plt.title('{} - Precision-recall curve'.format(clf_name))
    plt.plot(recall, precision, 'r', label='Precision-recall UnSMOTEd')
    plt.plot(recall_smote, precision_smote, 'g', label='Precision-recall SMOTEd')
    plt.legend(loc='upper right')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig('imbalance_plots/{}_Prec_Rec.png'.format(clf_name), bbox_inches='tight')


def string_to_timestamp(date_string):  # convert time string to float value
    time_stamp = time.strptime(date_string, '%Y-%m-%d %H:%M:%S')
    return time.mktime(time_stamp)


def make_clf(usx, usy, clf, clf_name, normalize=False, smoted=False):
    print('----------{}----------'.format(clf_name))
    totalTP, totalFP, totalFN, totalTN = 0, 0, 0, 0
    total_y_test = []
    total_y_prob = []
    total_y_pred = []
    skf = StratifiedKFold(n_splits=10)
    for train_index, test_index in skf.split(usx, usy):
        x_train, x_test = usx[train_index], usx[test_index]
        y_train, y_test = usy[train_index], usy[test_index]

        if smoted:
            sm = SMOTE(sampling_strategy=0.5)
            x_train, y_train = sm.fit_resample(x_train, y_train)

        if normalize:
            scaler = RobustScaler().fit(x_train)
            x_train = scaler.transform(x_train)
            x_test = scaler.transform(x_test)

        clf.fit(x_train, y_train)
        y_predict = clf.predict(x_test)
        y_proba = clf.predict_proba(x_test)
        # needed for roc curve
        total_y_test += list(y_test)
        total_y_prob += list(y_proba[:, 1])
        total_y_pred += list(y_predict)

        for i in range(len(y_predict)):
            if y_predict[i] and y_proba[i, 1] <= 0.65:
                y_predict[i] = 0
        for i in range(len(y_predict)):
            if y_test[i] and y_predict[i]:
                totalTP += 1
            if not y_test[i] and y_predict[i]:
                totalFP += 1
            if y_test[i] and not y_predict[i]:
                totalFN += 1
            if not y_test[i] and not y_predict[i]:
                totalTN += 1

    print('TOTAL TP: ' + str(totalTP))
    print('TOTAL FP: ' + str(totalFP))
    print('TOTAL FN: ' + str(totalFN))
    print('TOTAL TN: ' + str(totalTN))

    total_y_test = np.array(total_y_test)
    total_y_prob = np.array(total_y_prob)
    total_y_pred = np.array(total_y_pred)
    fpr, tpr, _ = metrics.roc_curve(total_y_test, total_y_prob)
    precision, recall, _ = precision_recall_curve(total_y_test, total_y_prob)
    roc_auc = metrics.auc(fpr, tpr)
    if smoted:
        plot_confusion_matrix(total_y_test, total_y_pred, clf_name, ['benign', 'fraudulent'], ' SMOTEd',
                            normalize=True, title=None, cmap=plt.cm.Blues)
    else:
        plot_confusion_matrix(total_y_test, total_y_pred, clf_name, ['benign', 'fraudulent'], '',
                              normalize=True, title=None, cmap=plt.cm.Blues)
    return fpr, tpr, roc_auc, precision, recall


if __name__ == "__main__":
    filename = 'original_data.csv'
    data = pd.read_csv(filename)
    x = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    x = np.delete(x, [2, 4, 5, 6, 7, 9, 11, 13], 1)
    clfs = {'KNeighborsClassifier': neighbors.KNeighborsClassifier(n_neighbors=3, algorithm='auto', weights='distance')
            , 'LogisticRegression': LogisticRegression(solver='newton-cg')
            , 'NaiveBayes': GaussianNB()
            , 'AdaBoostClassifier': AdaBoostClassifier(n_estimators=50)
            , 'RandomForestClassifier': RandomForestClassifier(n_estimators=100)
            , 'ExtraTreesClassifier': ExtraTreesClassifier(n_estimators=100)
            , 'GradientBoostingClassifier': GradientBoostingClassifier(n_estimators=100)
            , 'VotingClassifier': VotingClassifier(estimators=[
                    ('knn', neighbors.KNeighborsClassifier(n_neighbors=3, algorithm='kd_tree', weights='distance')),
                    ('lr', LogisticRegression(solver='newton-cg')),
                    ('gnb', GaussianNB())
                    ], voting='soft')
            }
    for clf_name, clf in clfs.items():
        usx = np.copy(x)
        usy = np.copy(y)
        if clf_name == 'LogisticRegression':
            fpr, tpr, roc_auc, precision, recall = make_clf(usx, usy, clf, clf_name, normalize=True)
            fpr_smote, tpr_smote, roc_auc_smote, precision_smote, recall_smote = \
                make_clf(usx, usy, clf, clf_name, normalize=True, smoted=True)
        else:
            fpr, tpr, roc_auc, precision, recall = make_clf(usx, usy, clf, clf_name)
            fpr_smote, tpr_smote, roc_auc_smote, precision_smote, recall_smote = \
                make_clf(usx, usy, clf, clf_name, smoted=True)
        plot_roc(fpr, tpr, roc_auc, fpr_smote, tpr_smote, roc_auc_smote, clf_name)
        # plot_prec_rec(precision, recall, precision_smote, recall_smote)
