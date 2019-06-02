import math
import numpy as np

# the breakpoints of the SAX method dependent on the alphabet length
breakpoints = {'3': [-0.43, 0.43],
               '4': [-0.67, 0, 0.67],
               '5': [-0.84, -0.25, 0.25, 0.84],
               '6': [-0.97, -0.43, 0, 0.43, 0.97],
               '7': [-1.07, -0.57, -0.18, 0.18, 0.57, 1.07],
               '8': [-1.15, -0.67, -0.32, 0, 0.32, 0.67, 1.15],
               '9': [-1.22, -0.76, -0.43, -0.14, 0.14, 0.43, 0.76, 1.22],
               '10': [-1.28, -0.84, -0.52, -0.25, 0, 0.25, 0.52, 0.84, 1.28],
               '11': [-1.34, -0.91, -0.6, -0.35, -0.11, 0.11, 0.35, 0.6, 0.91, 1.34],
               '12': [-1.38, -0.97, -0.67, -0.43, -0.21, 0, 0.21, 0.43, 0.67, 0.97, 1.38],
               '13': [-1.43, -1.02, -0.74, -0.5, -0.29, -0.1, 0.1, 0.29, 0.5, 0.74, 1.02, 1.43],
               '14': [-1.47, -1.07, -0.79, -0.57, -0.37, -0.18, 0, 0.18, 0.37, 0.57, 0.79, 1.07, 1.47],
               '15': [-1.5, -1.11, -0.84, -0.62, -0.43, -0.25, -0.08, 0.08, 0.25, 0.43, 0.62, 0.84, 1.11, 1.5],
               '16': [-1.53, -1.15, -0.89, -0.67, -0.49, -0.32, -0.16, 0, 0.16, 0.32, 0.49, 0.67, 0.89, 1.15, 1.53],
               '17': [-1.56, -1.19, -0.93, -0.72, -0.54, -0.38, -0.22, -0.07, 0.07, 0.22, 0.38, 0.54, 0.72, 0.93, 1.19,
                      1.56],
               '18': [-1.59, -1.22, -0.97, -0.76, -0.59, -0.43, -0.28, -0.14, 0, 0.14, 0.28, 0.43, 0.59, 0.76, 0.97,
                      1.22, 1.59],
               '19': [-1.62, -1.25, -1, -0.8, -0.63, -0.48, -0.34, -0.2, -0.07, 0.07, 0.2, 0.34, 0.48, 0.63, 0.8, 1,
                      1.25, 1.62],
               '20': [-1.64, -1.28, -1.04, -0.84, -0.67, -0.52, -0.39, -0.25, -0.13, 0, 0.13, 0.25, 0.39, 0.52, 0.67,
                      0.84, 1.04, 1.28, 1.64]
               }


def alphabetize(paa_data, alphabet_size):
    """
    Function that convert each value of a PAA segment into a discrete value given the above breakpoints
    :param paa_data: the PAA segments
    :param alphabet_size: the length of the alphabet
    :return: the discretized sequence
    """
    alphabetized = ''
    beta = breakpoints[str(alphabet_size)]
    for i in range(0, len(paa_data)):
        letter = False
        for j in range(0, len(beta)):
            if paa_data[i] < beta[j]:
                alphabetized += str(j)
                letter = True
                break
        if not letter:
            alphabetized += str(len(beta))

    return alphabetized


def PAA(x, word_size):
    """
    Function that performs the PAA dimensionality reduction
    :param x: the initial time series
    :param word_size: the number of PAA segments to produce at the end
    :return: the PAA segments along with the blocks that each segment represents
    """
    n = len(x)
    step_float = n / float(word_size)  # the float number of steps that are going to be compressed in each segment
    step = int(math.ceil(step_float))  # and then it is turned into integer
    chunk_start = 0
    approximated = []
    indices = []
    i = 0
    while chunk_start <= n - step:
        chunk = np.array(x[chunk_start:int(chunk_start + step)])  # take each chunk of time steps
        approximated.append(np.mean(chunk))  # and calculate the mean
        indices.append([chunk_start, int(chunk_start + step)])  # and append the initial indices of the segment
        i += 1
        chunk_start = int(i * step_float)  # the start of the next segment
    return np.array(approximated), indices


def to_letter_rep(data, word_size, alphabet_size):
    """
    Function that discretizes a data sequence using PAA dimensionality reduction and the SAX method
    :param data: the sequential input data
    :param word_size: the number of PAA segments to be produced
    :param alphabet_size: the length of the alphabet
    :return: the discretized signal along with the block indices of each PAA segment
    """
    (paa_data, indices) = PAA(normalize(data), word_size)
    return alphabetize(paa_data, alphabet_size), indices


def normalize(data, eps=1e-6):
    """
    Function that performs normalization on the data
    :param data: the initial data
    :param eps: a threshold set for the std of the data
    :return: the normalized data
    """
    x = np.asanyarray(data)
    if x.std() < eps:
        return [0]*len(x)
    else:
        return (x - x.mean()) / x.std()
