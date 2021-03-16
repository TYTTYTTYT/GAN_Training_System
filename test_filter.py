# Using a MLP structure and Wasserstein Metric

import time
import sys
import numpy as np
from torch import nn
from torch import optim
import torch
import torch.nn.functional as F
from data_reader import get_trajectory
from config import set_trim_length
import config
from data_analyzer import average_cca_score
from data_analyzer import average_spectra_diff_score
from data_analyzer import average_spectra_cca_score
from data_processor import fft_all_data
from data_processor import fft_data
from data_processor import ifft_data
from data_processor import pca_data
from data_processor import standardize_all_data
from data_processor import flatten_complex_data
from data_processor import iflatten_complex_data
from data_processor import time_stamp
from data_processor import trim_data
from dynamic_reporter import init_dynamic_report
from dynamic_reporter import stop_dynamic_report
from dynamic_reporter import report
from data_reader import write_one_file
from multiprocessing import set_start_method
from data_processor import standardize_data
from data_processor import istandardize_data_with
import random
import os

# Prepare the training set for this model
print('Preparing the training set...')
if config.TRIM_LENGTH is None:
    set_trim_length(300)
origin = trim_data(standardize_all_data())
data = fft_all_data()
train_set = flatten_complex_data(data)
print('Training set is ready!')

spectra = np.absolute(np.concatenate(data, axis=1))**2
print(spectra.shape)
total = np.sum(np.sum(np.concatenate(spectra, axis=0)))

print(total)


for i in range(spectra.shape[0]):
    print(i)
    print(np.sum(np.sum(spectra[:i, :])) / total)
