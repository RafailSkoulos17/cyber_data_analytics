from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from pca.utils import read_datasets


def plot_cumulative_explained_var(dataset):
    """
    Plots the cumulative explained variance over the number of principal components
    :param dataset: Dataset to apply PCA
    """
    # apply pca for the maximum number of features
    pca = PCA(n_components=dataset.shape[1])
    pca.fit(dataset)

    # compute cumulative explained variance as the components are increasing
    total_var = pca.explained_variance_ratio_.cumsum()

    # plot cumulative variance
    x_axis = np.arange(1, (len(pca.explained_variance_ratio_) + 1), 1)
    plt.title('Cumulative Variance of Principal Components')
    plt.xlabel('Principal Component')
    plt.ylabel('Cumulative Explained Variance')
    plt.plot(x_axis, total_var)
    plt.tight_layout()
    plt.grid()
    plt.savefig('../plots/pca/pca_var.png', bbox_inches='tight')
    # plt.show()


def get_num_of_components(dataset, explained_var):
    """
    Finds the minimum number of principal components that provide the given explained variance
    :param dataset: Dataset to apply PCA
    :param explained_var: Lower bound of explained variance we want
    :return: minimum number of principal components
    """
    # apply pca for the maximum number of features

    pca = PCA(n_components=31)
    pca.fit(dataset)

    # compute cumulative explained variance as the components are increasing
    total_var = pca.explained_variance_ratio_.cumsum()
    # find minimum number of principal components that provide the given explained variance
    n_components = [i + 1 for i, var in enumerate(total_var) if var > explained_var][0]
    return n_components


if __name__ == '__main__':
    # read all datasets
    scaled_df1, train_y1, scaled_df2, train_y2, scaled_test_df, y = read_datasets()

    # coluns to drop, status signals
    drop_columns = ['S_PU1', 'S_PU2', 'S_PU3', 'S_PU4', 'S_PU5', 'S_PU6', 'S_PU7', 'S_PU8', 'S_PU9', 'S_PU10', 'S_PU11',
                    'S_V2']

    # drop columns on all the 3 datasets
    pca_train_data1 = scaled_df1.select_dtypes(include=['float64']).drop(drop_columns, axis=1)
    pca_train_data2 = scaled_df2.select_dtypes(include=['float64']).drop(drop_columns, axis=1)
    pca_test_data = scaled_test_df.select_dtypes(include=['float64']).drop(drop_columns, axis=1)

    plot_cumulative_explained_var(pca_train_data1)
    print('Number of components for explained variance {0} is {1}'.format(0.99, get_num_of_components(pca_train_data1,
                                                                                                      explained_var=0.99)))
