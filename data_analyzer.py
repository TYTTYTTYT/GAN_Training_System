from data_reader import get_trajectory
import config
import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from scipy.fft import fft
import math

def cca_score(X, Y):
    # Calculate the CCA score of the first component pair
    ca = CCA(n_components=1)
    ca.fit(X, Y)
    Xc, Yc = ca.transform(X, Y)
    score = np.corrcoef(Xc[:, 0], Yc[:, 0])

    return score[0][1]

def average_cca_score(X, data):
    cca_array = np.empty(len(data))
    for i in range(len(data)):
        cca_array[i] = cca_score(X, data[i])

    score = cca_array.mean()
    if score == np.nan:
        score = 0
    
    return score

def data_mean():
    if config.DATA_MEAN is not None:
        return config.DATA_MEAN

    data = get_trajectory()

    data = np.concatenate(data, axis=0)
    mean = data.mean(axis=0)

    config.DATA_MEAN = mean

    return mean

def data_std():
    if config.DATA_STD is not None:
        return config.DATA_STD

    data = get_trajectory()

    data = np.concatenate(data, axis=0)
    std = data.std(axis=0)

    config.DATA_STD = std

    return std

def data_length(data):
    # Calculate the total length of a data set
    length = 0
    for d in data:
        length += len(d)

    return length

def data_pca(n_cp=None):
    if config.PCA is not None:
        return config.PCA

    if config.STANDARDIZED is False:
        raise Exception("Standardize data before PCA!")

    data = get_trajectory()
    X = np.concatenate(data, axis=0)

    pca = PCA(n_components=n_cp)
    pca.fit(X)

    config.PCA = pca

    return pca

def spectra_cca(X, Y):
    # CCA score on sqrt(power)
    X = np.absolute(X).astype(np.float32)
    Y = np.absolute(Y).astype(np.float32)

    return cca_score(X, Y)

def spectra_diff(X, Y):
    # Averaged Euclidean distance on sqrt(power)
    X = np.absolute(X)
    Y = np.absolute(Y)

    distance = np.sum(np.absolute((X - Y)), axis=0)

    return float(np.sum(distance) / X.shape[1] / X.shape[0])

    # Average difference on all values
    Xa = np.absolute(X)
    Ya = np.absolute(Y)

    diff = np.sum(np.sum(np.absolute(Xa - Ya)))
    diff = float(diff)

    return diff / Xa.shape[0] / Xa.shape[1]

def average_spectra():
    if config.AVERAGE_SPECTRA is not None:
        return config.AVERAGE_SPECTRA
    if config.ALL_COMPLEX_DATA is None:
        raise Exception("fft_all_data before invoke this function!")

    data = config.ALL_COMPLEX_DATA
    spectra = np.zeros_like(np.absolute(data[0]))

    for d in data:
        spectra += np.absolute(d)

    spectra = spectra / len(data)
    config.AVERAGE_SPECTRA = spectra

    return spectra

def average_spectra_diff(X):
    diff = spectra_diff(X, average_spectra())

    return diff

def average_spectra_diff_score(X):
    diff = average_spectra_diff(X)
    score = 1 - math.tanh(diff / 2)

    return score

def average_spectra_cca_score(X):
    score = spectra_cca(X, average_spectra())

    return score
