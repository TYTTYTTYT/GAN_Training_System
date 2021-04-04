# %%
from data_reader import get_trajectory
from data_reader import _read_one_file
from data_processor import fft_data
from data_processor import ifft_data
from config import set_trim_length
from data_processor import standardize_data
from data_processor import standardize_all_data
from data_processor import trim_data
from data_processor import window
from data_processor import lpf_dimension_reduction
from data_processor import pad_data_zeros
from data_processor import iwindow
from data_processor import data_center_part
import numpy as np
import matplotlib.pyplot as plt
from data_processor import low_pass_filter
from data_analyzer import get_single_side_frequency
from data_analyzer import data_mean
from data_analyzer import data_std
import random
from model_complex_fullconnected import Complex_Fully_Connected_GAN
from model_complex_fully_connected_wgan import Complex_Fully_Connected_WGAN
from model_wgan_lpf_window import Complex_Fully_Connected_WGAN_LPF_W
from model_complex_fully_connected_wgan_lpf import Complex_Fully_Connected_WGAN_LPF
from play import get_real_example
import play
from play_samples import Choosed_MLP
from play_samples import Choosed_WGAN
import torch

def load_model(model, path, args):
    net = model(*args)
    net.load_state_dict(torch.load(path))

    return net

def sample_net(net, n):
    samples = []
    for i in range(n):
        sample = get_real_example(net)
        sample, _, _ = standardize_data(sample, mean=data_mean(), std=data_std())
        samples.append(sample)

    return samples

def get_random_example(n, trim_length, trimed=True, standardize=True):
    if standardize:
        data = standardize_all_data()
    else:
        data = get_trajectory()

    if trimed:
        data = trim_data(data, length=trim_length)

    examples = random.choices(data, k=n)

    return examples

def sqrt_spectra(examples):
    n = examples[0].shape[0]
    freq = get_single_side_frequency(n)
    
    spectras = []
    fe = fft_data(examples)
    for e in fe:
        spectras.append((freq, np.absolute(e)))

    return spectras

def log_spectra(examples):
    n = examples[0].shape[0]
    freq = get_single_side_frequency(n)
    
    spectras = []
    fe = fft_data(examples)
    for e in fe:
        m = np.amax(np.absolute(e))
        spectras.append((freq, 20 * np.log10(np.absolute(e) / m)))

    return spectras

def draw_frequency_analysis(exampes):
    N = len(exampes)
    spectras = sqrt_spectra(exampes)
    log_spectras = log_spectra(exampes)
    figure, axs = plt.subplots(N, 3, figsize=(20, 45))
    for i in range(N):
        ax = axs[i]
        ax[0].set_title('Spectra')
        ax[1].set_title('Spectra')
        ax[2].set_title('Standardized Trajectory')

        ax[0].set_ylabel('Sqrt power')
        ax[1].set_ylabel('Power (dB)')
        ax[0].set_xlabel('Frequency(Hz)')
        ax[1].set_xlabel('Frequency(Hz)')
        ax[2].set_xlabel('Frame')

        ax[0].plot(*spectras[i])
        ax[1].plot(*log_spectras[i])
        ax[2].plot(exampes[i])

    plt.show()

    return

def draw_frequency_analysis_lpf(examples):
    N = len(examples)
    examples_f = []
    for i in range(N):
        examples_f.append(low_pass_filter(examples[i]))
    spectras = sqrt_spectra(examples_f)
    log_spectras = log_spectra(examples_f)
    figure, axs = plt.subplots(N, 3, figsize=(20, 45))
    for i in range(N):
        ax = axs[i]
        ax[0].set_title('Spectra')
        ax[1].set_title('Spectra')
        ax[2].set_title('Standardized trajectory')

        ax[0].set_ylabel('Sqrt power')
        ax[1].set_ylabel('Log power')
        ax[0].set_xlabel('Frequency(Hz)')
        ax[1].set_xlabel('Frequency(Hz)')
        ax[2].set_xlabel('Frame')

        ax[0].plot(*spectras[i])
        ax[1].plot(*log_spectras[i])
        ax[2].plot(examples_f[i])

    plt.show()

    return

def lpf_compare(examples):
    N = len(examples)
    examples_lpf = []
    for e in examples:
        examples_lpf.append(low_pass_filter(e))

    figure, axs = plt.subplots(N, 2, figsize=(15, 45))

    for i in range(N):
        ax = axs[i]

        ax[0].set_title('Standardized trajectory')
        ax[1].set_title('Filtered trajectory')

        ax[0].set_xlabel('Frame')
        ax[1].set_xlabel('Frame')

        ax[0].plot(examples[i])
        ax[1].plot(examples_lpf[i])

    plt.show()

    return

def compare(*argv):
    N = len(argv)
    length = len(argv[0])

    figure, axs = plt.subplots(length, N, figsize=(15, 45))

    for i in range(length):
        ax = axs[i]

        for j in range(N):
            ax[j].plot(argv[j][i])

# %%
examples = get_random_example(6, 1000)
# %%
windowed, win = window(examples, 'flattop')
# %%
draw_frequency_analysis(windowed)

# %%
draw_frequency_analysis(examples)
# %%
draw_frequency_analysis_lpf(examples)

# %%
lpf_compare(examples)
# %%
MLP = load_model(Complex_Fully_Connected_GAN, Choosed_MLP, (6, ))
# %%
samples = sample_net(MLP, 6)
# %%
draw_frequency_analysis(samples)
# %%
draw_frequency_analysis_lpf(samples)

# %%
lpf_compare(samples)
# %%
WGAN = load_model(Complex_Fully_Connected_WGAN, Choosed_WGAN, (6, ))
# %%
samples = sample_net(WGAN, 6)
# %%
draw_frequency_analysis(samples)
# %%
draw_frequency_analysis_lpf(samples)

# %%
lpf_compare(samples)
# %%

origin = get_random_example(6, 300)
originf = fft_data(origin)
yf, _ = lpf_dimension_reduction(originf, 5)
yf = pad_data_zeros(yf, 151)
y = ifft_data(yf)
# %%
draw_frequency_analysis(origin)
# %%
draw_frequency_analysis(y)
# %%
compare(origin, y)
# %%
originw, win = window(origin, 'hamming')
originfw = fft_data(originw)
yfw, _ = lpf_dimension_reduction(originfw, 5)
yfw = pad_data_zeros(yfw, 151)
yw = ifft_data(yfw)
yw = iwindow(yw, win, 0.7)
# %%
draw_frequency_analysis(origin)
# %%
draw_frequency_analysis(y)
# %%
compare(data_center_part(origin, 0.7), yw)

# %%
compare(data_center_part(origin, 0.7), data_center_part(y, 0.7))
# %%
compare(data_center_part(origin, 0.7), data_center_part(y, 0.7), yw)

# %%
