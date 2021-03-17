# This module is used to play videos from model
import os
from data_reader import write_one_file
from data_processor import istandardize_data
from data_processor import overlap_and_add
import torch
import config

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
