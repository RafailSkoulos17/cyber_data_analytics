from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from anomaly_detection.pca.tune_pca import get_num_of_components, get_threshold
from utils import read_datasets, get_score, plot_anomalies
import pickle
import warnings
warnings.filterwarnings("ignore")


def pca_detect():
    # read training 1 dataset
    scaled_df1, train_y1, scaled_df2, train_y2, scaled_test_df, y = read_datasets()

    # coluns to drop, status signals
    drop_columns = ['S_PU1', 'S_PU2', 'S_PU3', 'S_PU4', 'S_PU5', 'S_PU6', 'S_PU7', 'S_PU8', 'S_PU9', 'S_PU10', 'S_PU11',
                    'S_V2']

    # drop columns on all the 3 datasets
    pca_train_data1 = scaled_df1.drop(drop_columns, axis=1)
    pca_train_data2 = scaled_df2.drop(drop_columns, axis=1)
    pca_test_data = scaled_test_df.drop(drop_columns, axis=1)

    # apply PCA on training dataset 1
    components = get_num_of_components(pca_train_data1, 0.99)  # determine the number of PCA components
    pca = PCA(n_components=components)
    pca.fit(pca_train_data1)
    # compute residuals
    residuals = pca_train_data1 - pca.inverse_transform(pca.transform(pca_train_data1))
    res_norm = np.sqrt(np.square(residuals).sum(axis=1))
    # find threshold for outlier detection => mean + std
    threshold = np.mean(res_norm) + 3* np.std(res_norm)  # increase std*(2 or 3), increases TP by 30 and FP by 50

    print('Threshold is {}'.format(threshold))
    # find outliers
    predicted_positives = np.where(res_norm > threshold)[0]
    # uncomment to plot residuals and point outliers
    plt.plot(res_norm)
    thresh_handle = plt.plot_date([res_norm.index[0], res_norm.index[-1]], [threshold, threshold], fmt='-', color='r')
    plt.ylabel("Square Prediction Error")
    plt.xlabel("Observation")
    plt.title("Application of PCA to training data with " + str(components) + " normal components")
    plt.legend([thresh_handle], ["Sample threshold"])
    plt.savefig('../plots/pca/train_outliers.png', bbox_inches='tight')
    # plt.show()
    print("Shape before is {}".format(str(pca_train_data1.shape)))
    pca_train_data1_without_outliers = pca_train_data1.drop(pca_train_data1.index[predicted_positives])
    print("Shape after is {}".format(str(pca_train_data1_without_outliers.shape)))

    # we use training dataset 2 to find the optimal classification threshold
    pca = PCA(n_components=pca_train_data2.shape[1])
    pca.fit(pca_train_data1_without_outliers)
    components_dataset2 = get_num_of_components(pca_train_data1_without_outliers, 0.95)
    threshold = get_threshold(pca, components_dataset2, conf=0.95)

    # another way to get a threshold
    # residuals = pca_train_data2 - pca.inverse_transform(pca.transform(pca_train_data2))
    # res_norm = np.sqrt(np.square(residuals).sum(axis=1))
    # total_errors = [err for i, err in enumerate(res_norm) if train_y2[i] == 1]
    # threshold = np.max(total_errors) # computed classification threshold

    # apply PCA for test dataset
    pca = PCA(n_components=components)
    pca.fit(pca_train_data1_without_outliers)
    residuals = pca_test_data - pca.inverse_transform(pca.transform(pca_test_data))
    res_norm = np.sqrt(np.square(residuals).sum(axis=1))
    # find indices of predicted and true positives
    predicted_anomalies = np.where(res_norm > threshold)[0]
    true_anomalies = np.where(y > 0)[0]

    # compute scores
    [tp, fp, fn, tn, tpr, tnr, Sttd, Scm, S] = get_score(predicted_anomalies, true_anomalies, y=y)

    print("TP: {0}, FP: {1}, TPR: {2}, TNR: {3}".format(tp, fp, tpr, tnr))
    print("Sttd: {0}, Scm: {1}, S: {2}".format(Sttd, Scm, S))

    # plot residuals and threshold line
    plt.figure()
    plt.plot(res_norm)
    thresh_handle, = plt.plot_date([res_norm.index[0], res_norm.index[-1]], [threshold, threshold], fmt='-', color='r')
    plt.ylabel("Square Prediction Error")
    plt.xlabel("Observation")
    plt.title("PCA with " + str(components) + " components")
    plt.legend([thresh_handle], ["Threshold"])
    plt.savefig('../plots/pca/residuals_test.png', bbox_inches='tight')
    # plt.show()

    # plot attacks detected
    all_predictions = [1 if res > threshold else 0 for res in res_norm]
    plot_anomalies(y, all_predictions)
    return predicted_anomalies


if __name__ == '__main__':
    predicted_anomalies = pca_detect()
    with open('../pca_all.pickle', 'wb') as handle:
        pickle.dump(predicted_anomalies, handle, protocol=pickle.HIGHEST_PROTOCOL)
