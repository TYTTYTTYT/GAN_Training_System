# This module is used to play videos from model
import os
from data_reader import write_one_file
from data_processor import istandardize_data
from data_processor import overlap_and_add
import matplotlib.pyplot as plt
import torch
import config
import random
import numpy as np
from data_reader import _read_one_file
from data_processor import standardize_data
from data_analyzer import data_mean
from data_analyzer import data_std
from data_processor import fft_data
from data_processor import trim_data
from data_processor import griffin

CMD = os.path.join(
    config.GALATEA_PATH,
    'Hiroshi/MotionCapture/Export/gRigidBodies4'
)
PAR1 = " -width 1000 -height 1000 -grid -axes -zoom 4 -loop -head_offset_angles 0 0 -3 -origin -0.3 0.52 0.18 -head_offset_position -0.1 -0.1 0 -rb_mag 0.0110 -axis_mag 0.01 -volume 1.0 -head "
PAR2 = " -show_head "
TRANSLATION = "-translation"

def get_real_example(net):
    example = [net.example()]
    example = istandardize_data(example)[0]

    return example

def play_one_video_from(model, path, format='rov', args=None, translation=True):
    net = model(*args)

    net.load_state_dict(torch.load(path))
    
    example = [net.example()]
    example = istandardize_data(example)[0]

    write_one_file('example', example, format=format)

    cmd = CMD + PAR1 + 'example' + PAR2
    if translation:
        cmd += TRANSLATION

    os.system(cmd)

    return net

def play_long_video_from(model, path, length, format='rov', args=None, translation=True):
    net = model(*args)
    net.load_state_dict(torch.load(path))

    y = get_real_example(net)
    while y.shape[0] < length:
        y = overlap_and_add(y, get_real_example(net))

    write_one_file('example', y, format=format)

    cmd = CMD + PAR1 + 'example' + PAR2
    if translation:
        cmd += TRANSLATION

    os.system(cmd)

    return y

def random_log_spetra_from(data):
    x = random.choice(data)

    plt.plot(np.log(x))
    plt.show()

    return

def draw_spectra_of(path):
    x = _read_one_file(path)
    x, _, _ = standardize_data(x, data_mean(), data_std())
    x = fft_data([x])[0]

    x = np.absolute(x)

    plt.plot(x)
    plt.show()

    return x

def draw_log_spectra_of(path):
    x = _read_one_file(path)
    x, _, _ = standardize_data(x, data_mean(), data_std())
    x = fft_data([x])[0]

    x = np.absolute(x)
    x = np.log(x)

    plt.plot(x)
    plt.show()

    return x

def draw_trajectory_of(path):
    x = _read_one_file(path)

    plt.plot(x)
    plt.show()

    return x

def play_long_video_istft(model, path, n, win, format='rov', args=None, translation=True):
    net = model(*args)
    net.load_state_dict(torch.load(path))

    data = []
    for i in range(n):
        data.append(net.example()[0])

    print('data ' + str(len(data)))

    result = griffin(data, win)
    result = istandardize_data([result])[0]

    write_one_file('example', result, format=format)

    cmd = CMD + PAR1 + 'example' + PAR2
    if translation:
        cmd += TRANSLATION

    os.system(cmd)

    return result
