import config
import numpy as np
from data_reader import get_trajectory
from data_analyzer import data_mean
from data_analyzer import data_std
from data_analyzer import data_pca
from scipy.fft import fft, ifft
import datetime
import types

def low_pass_filter(data, f):
    n = len(data)
    for i in range(n):
        data[i] = data[i][:f, :]

    return data

def pad_zeros(x, n):
    rows = x.shape[0]
    columns = x.shape[1]

    y = np.zeros((n, columns), dtype=np.float32)
    for i in range(rows):
        y[i, :] = x[i, :]

    return y

def pad_data_zeros(data, n):
    for i in range(len(data)):
        data[i] = pad_zeros(data[i], n)

    return data


def standardize_data(data):
    if isinstance(data, list):
        data = np.concatenate(data, axis=0)

    mean = data.mean(axis=0)
    std = data.std(axis=0)
    print(mean)
    print(std)

    data = (data - mean)
    for i in range(std.shape[0]):

        if not np.isclose(std[i], 0.):
            data[i, :] = data[i, :] / std[i]

    return data, mean, std

def istandardize_data_with(data, mean, std):
    data = data * std + mean

    return data

def standardize_all_data():
    # Standardize all data
    # Let data's mean = 0
    # standardard deviation = 1
    # If data are not loaded, this function will load data first
    # Once all data are standardized,
    # function get_trajectory will return standardized data
    if config.STANDARDIZED is True:
        return get_trajectory()
    if config.ALL_TRAJECTORY is None:
        # Load data set
        get_trajectory()

    data = config.ALL_TRAJECTORY
    mean = data_mean()
    std = data_std()

    for i in range(len(data)):
        data[i] = list(data[i])
        # First two elements are matadata, 3rd is data
        data[i][3] = (data[i][3] - mean) / std
        data[i] = tuple(data[i])

    config.ALL_TRAJECTORY = data
    config.STANDARDIZED = True

    return get_trajectory()

def istandardize_data(data):
    # Recover standardized data
    # The parameter 'data' should be a list of real numpy array
    # element in data has shape length * dimensionality
    if config.STANDARDIZED is False:
        raise Exception('Data set has not been standardized!')

    mean = data_mean()
    std = data_std()
    recovered = []

    for d in data:
        r = d * std + mean
        recovered.append(d)

    return recovered

def trim_data(data, length=None):
    if length is not None:
        config.TRIM_LENGTH = length
    elif config.TRIM_LENGTH is None:
        raise Exception("Trim length has not been setted up!")
    
    length = config.TRIM_LENGTH
    trimed = []

    for d in data:
        n = int(len(d) / length)
        idx = 0
        for i in range(n):
            trimed.append(d[idx:idx + length])
            idx += length

    return trimed

def pca_data(data):
    pca = data_pca()

    n = len(data)
    for i in range(n):
        data[i] = pca.fit_transform(data[i])
    
    return data

def ipca_data(data):
    pca = data_pca()

    n = len(data)
    for i in range(n):
        data[i] = pca.inverse_transform(data[i])

    return data


def fft_data(data):
    # Apply DFT on each data and each dimension,
    # Data should be trimed first
    # Data should be even
    n = len(data)
    N = data[0].shape[0]
    w = data[0].shape[1]

    data_f = []
    for i in range(n):
        f = np.empty((N // 2 + 1, w), dtype=np.complex128)
        for j in range(w):
            f[:, j] = fft(np.array(data[i])[:, j])[:N // 2 + 1]
        
        data_f.append(f)

    return data_f

def ifft_data(data_f):
    n = len(data_f)
    N = data_f[0].shape[0]
    w = data_f[0].shape[1]

    data = []
    for i in range(n):
        d = np.empty([(N - 1) * 2, w], dtype=np.float)
        for j in range(w):
            reverse = np.conj(np.flip(data_f[i][1:N - 1, j]))
            d[:, j] = ifft(np.concatenate([data_f[i][:, j], reverse])).real

        data.append(d)

    return data

def fft_all_data():
    if config.ALL_COMPLEX_DATA is not None:
        return config.ALL_COMPLEX_DATA
    if config.TRIM_LENGTH is None:
        raise Exception("Trim length has not been setted up!")

    trimed = trim_data(get_trajectory())
    trimedf = fft_data(trimed)
    config.ALL_COMPLEX_DATA = trimedf

    return trimedf

def flatten_complex_data(data):
    if config.TRIM_LENGTH is None:
        raise Exception("Trim data first!")

    n = len(data)
    length = data[0].shape[0]
    width = data[0].shape[1]

    flattened = np.empty([n, length * width * 2])
    
    for i in range(n):
        d = data[i]

        real = np.real(d)
        img = np.imag(d)

        real = real.reshape(1, length * width)
        img = img.reshape(1, length * width)

        df = np.concatenate([real, img], axis=1)

        flattened[i, :] = df
    
    return flattened

def iflatten_complex_data(flattened):
    data = []

    n = flattened.shape[0]
    length = config.TRIM_LENGTH // 2 + 1
    width = flattened.shape[1] // 2 // length

    for i in range(n):
        d = np.empty([length, width], dtype=np.complex128)
        df = flattened[i, :]

        real = df[:length * width]
        img = df[length * width:]

        real = real.reshape(length, width)
        img = img.reshape(length, width)

        d.real = real
        d.imag = img

        data.append(d)

    return data

def iflatten_complex_data_with(flattened, length):
    data = []

    n = flattened.shape[0]
    width = flattened.shape[1] // 2 // length

    for i in range(n):
        d = np.empty([length, width], dtype=np.complex128)
        df = flattened[i, :]

        real = df[:length * width]
        img = df[length * width:]

        real = real.reshape(length, width)
        img = img.reshape(length, width)

        d.real = real
        d.imag = img

        data.append(d)

    return data

def time_stamp():
    return '{:%Y-%m-%d|%H:%M:%S}'.format(datetime.datetime.now())
