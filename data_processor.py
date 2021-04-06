import config
import numpy as np
from data_reader import get_trajectory
from data_analyzer import data_mean
from data_analyzer import data_std
from data_analyzer import data_pca
from data_analyzer import get_single_side_frequency
from scipy.fft import fft, ifft
import datetime
import types
from scipy import signal


def pad_zeros(x, n):
    rows = x.shape[0]
    columns = x.shape[1]

    y = np.zeros((n, columns), dtype=x.dtype)
    for i in range(rows):
        y[i, :] = x[i, :]

    return y

def pad_data_zeros(data, n):
    for i in range(len(data)):
        data[i] = pad_zeros(data[i], n)

    return data


def standardize_data(data, mean=None, std=None):
    if isinstance(data, list):
        data = np.concatenate(data, axis=0)

    if mean is None:
        mean = data.mean(axis=0)
    if std is None:
        std = data.std(axis=0)
    print(mean)
    print(std)

    data = (data - mean) / std

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
        recovered.append(r)

    return recovered

def trim_data(data, length=None):
    if length is not None and config.TRIM_LENGTH is None:
        config.TRIM_LENGTH = length
    elif config.TRIM_LENGTH is None and length is None:
        raise Exception("Trim length has not been setted up!")
    elif length is None:
        length = config.TRIM_LENGTH
    
    trimed = []

    for d in data:
        n = int(len(d) / length)
        idx = 0
        for i in range(n):
            trimed.append(d[idx:idx + length])
            idx += length

    return trimed

def pca_data(data, n):
    pca = data_pca(n)

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

def overlap_and_add(x1, x2):
    l1 = x1.shape[0]
    l2 = x2.shape[0]

    if l1 < l2:
        l_over = l1 // 2
    else:
        l_over = l2 // 2

    decrease = np.reshape(np.linspace(1, 0, num=l_over), (l_over, 1))
    increase = 1 - decrease

    x1_left = x1[:-l_over]
    x1_overlap = x1[-l_over:]
    x2_left = x2[l_over:]
    x2_overlap = x2[:l_over]

    overlap = x1_overlap * decrease + x2_overlap * increase

    y = np.concatenate([x1_left, overlap, x2_left], axis=0)

    return y

def low_pass_filter(example):
    x = np.copy(example)
    b, a = signal.butter(3, 0.1)
    x = np.array(x)
    y = np.empty_like(x)
    print(x.shape)
    for i in range(x.shape[1]):

        y[:, i] = signal.filtfilt(b, a, x[:, i])

    return y

def low_pass_filter_list(data):
    filtered = []
    for d in data:
        filtered.append(low_pass_filter(d))

    return filtered

def window(data, window_type):
    data_length = data[0].shape[0]

    window = np.reshape(signal.get_window(window_type, data_length), (data_length, 1))

    data_windowed = []

    for d in data:
        d = np.array(d)
        d = d * window
        data_windowed.append(d)

    return data_windowed, window

def iwindow(data, win, per=0.8):
    data_length = data[0].shape[0]
    idx_start = int(data_length * (1 - per) // 2)
    idx_end = int(data_length * (1 - (1 - per) / 2))

    data_recovered = []

    for d in data:
        r = d[idx_start:idx_end, :] / win[idx_start: idx_end]
        data_recovered.append(r)

    return data_recovered

def lpf_dimension_reduction(data, frequency, trim_length=None):
    if trim_length is None:
        trim_length = config.TRIM_LENGTH

    bins = data[0].shape[0]
    freq = get_single_side_frequency(trim_length)

    idx = freq <= frequency
    dim = np.sum(idx)

    data_reducted = []
    for d in data:
        data_reducted.append(d[idx])

    return data_reducted, dim

def data_center_part(data, per):
    center_parts = []
    data_length = data[0].shape[0]
    idx_start = int(data_length * (1 - per) // 2)
    idx_end = int(data_length * (1 - (1 - per) / 2))

    for d in data:
        r = d[idx_start:idx_end]
        center_parts.append(r)

    return center_parts

def griffin(data, win):
    data_windowed = []
    for d in data:
        data_windowed.append(np.array(d * win))

    print('awf ' + str(len(data_windowed)))

    half = data[0].shape[0] // 2
    win2 = win[:half]**2 + win[-half:]**2 + 0.001
    result = data_windowed[0]

    for d in data_windowed[1:]:
        result[-half:] += d[:half]
        result[-half:] = result[-half:] / win2
        result = np.concatenate((result, d[half:]))

    result = result[half + 1:-half - 1]

    return result
