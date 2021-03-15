# This module is used to play videos from model
import os
from data_reader import write_one_file
import torch
import config

CMD = os.path.join(
    config.GALATEA_PATH,
    'Hiroshi/MotionCapture/Export/gRigidBodies4'
)
PAR1 = " -width 500 -height 550 -grid -axes -zoom 4 -loop -head_offset_angles 0 0 -3 -origin -0.3 0.52 0.18 -head_offset_position -0.1 -0.1 0 -rb_mag 0.0110 -axis_mag 0.01 -volume 1.0 -head "
PAR2 = " -show_head "
TRANSLATION = "-translation"

def play_one_video_from(model, path, format='rov', args=None):
    net = model(*args)

    net.load_state_dict(torch.load(path))
    
    example = net.example()
    write_one_file('example', example, format=format)

    cmd = CMD + PAR1 + 'example' + PAR2

    os.system(cmd)

    return net
