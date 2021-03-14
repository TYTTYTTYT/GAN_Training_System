import config
from config import ACTORS
from config import DATA_PATH
import pandas as pd
import numpy as np
import os
import re
import pickle
import copy


def _read_one_file(directory):
    # read one file from any format in it's original format
    # No data pre-processing steps here
    # leave RMS values
    numbers = []
    order = ''
    with open(directory, "r") as file:
        in_header = True
        for line in file:
            if not in_header:
                numbers.append([float(i) for i in line.split()[0:-1]])

            if line.startswith("END_OF_HEADER"):
                in_header = False

            if line.startswith('ORDER'):
                order = line.split()[1:-1]

    trajectory = pd.DataFrame.from_records(numbers, columns=order)

    return trajectory

def _read_all_trajectory(format='rov', normalised=True):
    all_data = []
    if config.ALL_TRAJECTORY is not None:
        if config.DATA_FORMAT != format:
            raise Exception(
                "Wrong file format: {a}; existing file format: {b}".format(
                    a=format,
                    b=config.DATA_FORMAT
                )
            )
        return config.ALL_TRAJECTORY

    if normalised:
        n = 'n'
    else:
        n = 'r'
    stored_data = os.path.join(DATA_PATH, 'Data_Structures', 'all_trajectory_{f}_{n}'.format(f=format, n=n))
    if os.path.isfile(stored_data):
        with open(stored_data, 'rb') as f:
            all_data = pickle.load(f)
            config.ALL_TRAJECTORY = all_data
            config.DATA_LABELS = all_data[0][3].columns
            config.DATA_FORMAT = format

        return all_data

    if normalised:
        mid_path = '/Trajectory_Data/Normalised/'
    else:
        mid_path = '/Trajectory_Data/Raw/'

    for actor in ACTORS:
        fold = DATA_PATH + mid_path + actor
        stream = os.popen('ls {fold} | grep {actor}_.._..{format}'.format(fold=fold, actor=actor, format=format))
        file_names = stream.read().split()

        for f in file_names:
            path = fold + '/' + f
            trajectory = _read_one_file(path)
            info = f.split("_")
            all_data.append((info[0], info[2][0], int(info[1]), trajectory))

    with open(stored_data, 'wb') as f:
        pickle.dump(all_data, f)

    config.ALL_TRAJECTORY = all_data
    config.DATA_LABELS = all_data[0][3].columns
    config.DATA_FORMAT = format

    return all_data


def get_trajectory(name=None, emotion=None, format='rov', normalised=True, seq=None):
    if format not in config.FORMATS:
        raise Exception("Invalid file format: {}".format(format))
    if name is not None:
        if name not in config.ACTORS:
            raise Exception("Invalid actor name: {}".format(name))

    all_data = _read_all_trajectory(format=format, normalised=normalised)
    filtered = []

    for i in all_data:
        if name is not None:
            if i[0] != name:
                continue
        if emotion is not None:
            if i[1] != emotion:
                continue
        if seq is not None:
            if i[2] != seq:
                continue

        filtered.append(i[3])

    return filtered

def write_one_file(directory, trajectory, format='rov'):
    contents = config.HEADERS[format]

    rows = trajectory.shape[0]
    columns = trajectory.shape[1] + 1

    rms = np.zeros([rows, 1])
    data = np.concatenate([trajectory, rms], axis=1)

    for i in range(rows):
        for j in range(columns):
            contents += str(data[i, j])
            contents += ' '
        contents += '\n'

    with open(directory, 'w') as f:
        f.write(contents)

    return


if __name__ == "__main__":
    data = get_trajectory(name='Adam', seq=1)
    print(len(data))
    print(data[0])
