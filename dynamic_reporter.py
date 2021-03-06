# This module provides dynamic report functions,
# which can plot training loss and scores while training
# the data passed into QUEUE should be a dictionary with format
# {
#   loss_title:     str,
#   losses:         [[float]]
#   loss_labels:    [str],
#   score_title:    str,
#   scores:         [[float]],
#   score_labels:   list[str * n],
#   inteval:        int
#   example:        ndarray
# }
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Process
from multiprocessing import Queue
from matplotlib.animation import FuncAnimation
from data_processor import fft_data
import time
from data_analyzer import get_single_side_frequency

# Initialize local public variables
PROCESS = None
FIGURE = None
AX1 = None
AX2 = None
AX3 = None
AX4 = None
QUEUE = None
DATA = None

def report(loss_title, losses, loss_labels, score_title, scores, score_labels, interval, example):
    # Init the reporter first to report
    if 'INTERVAL' not in globals():
        INTERVAL = 0

    report = {
        'loss_title': loss_title,
        'losses': losses,
        'loss_labels': loss_labels,
        'score_title': score_title,
        'scores': scores,
        'score_labels': score_labels,
        'interval': interval,
        'example': example
    }

    if not QUEUE.empty():
        return

    QUEUE.put(report)

    return

def init_dynamic_report(interval, directory):
    global PROCESS
    global QUEUE
    global AX1
    global AX2
    global AX3
    global AX4

    QUEUE = Queue()
    figure, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(35, 35), num='Dynamic Report')
    plt.subplots_adjust(hspace=0.4)

    if PROCESS is not None:
        raise Exception('Stop other dynamic report first!')
    PROCESS = Process(
        target=_animation,
        args=(interval * 1000, figure, ax1, ax2, ax3, ax4, QUEUE, directory)
    )
    PROCESS.start()

    return

def stop_dynamic_report():
    global PROCESS
    global FIGURE
    global AX1
    global AX2
    global AX3
    global AX4
    global QUEUE
    global DATA

    if PROCESS is None:
        raise Exception('Dynamic report has not initialized!')
    QUEUE.put('close')
    PROCESS.join()
    
    PROCESS = None
    FIGURE = None
    AX1 = None
    AX2 = None
    AX3 = None
    AX4 = None
    QUEUE = None
    DATA = None

    return

def _animation(interval, figure, ax1, ax2, ax3, ax4, queue, directory):
    global FIGURE
    global QUEUE
    global AX1
    global AX2
    global AX3
    global AX4

    FIGURE = figure
    QUEUE = queue
    AX1 = ax1
    AX2 = ax2
    AX3 = ax3
    AX4 = ax4

    ani = FuncAnimation(
        fig=FIGURE,
        func=_draw,
        interval=interval
    )

    with open('/home/tai/Desktop/hello.txt', 'w') as f:
        f.write(str(ani))
    plt.show()
    try:
        FIGURE.savefig(directory)
    except ValueError:
        print('An error occured when saving the training report: ' + directory)
    else:
        print('File {} saved.'.format(directory))

    return

def _draw(i):
    global AX1
    global AX2
    global AX3
    global AX4
    global FIGURE
    global QUEUE
    global DATA

    if not QUEUE.empty():
        DATA = QUEUE.get()
        
        if DATA == 'close':
            plt.close(FIGURE)
            return

        example_length = DATA['example'].shape[0]
        freq = get_single_side_frequency(example_length)
        # DATA['loss_title'] = data['loss_title']
        # DATA['losses'].append(data['loss']),
        # DATA['loss_labels'] = data['loss_labels']
        # DATA['score_title'] = data['score_title']
        # DATA['scores'].append(data['score'])
        # DATA['score_labels'] = data['score_labels']
        # DATA['interval'] = data['interval']
        n = len(DATA['losses'])
        interval = DATA['interval']

        x = np.arange(interval, interval * n + 1, interval)
        loss_Y = np.array(DATA['losses']).T
        score_Y = np.array(DATA['scores']).T

        AX1.clear()
        AX2.clear()
        AX3.clear()
        AX4.clear()

        AX1.set_title(DATA['loss_title'])
        AX1.set_xlabel('Training Steps')
        AX1.set_ylabel('Model Loss')

        for i in range(loss_Y.shape[0]):
            AX1.plot(x, loss_Y[i].flatten(), label=DATA['loss_labels'][i])
        AX1.legend()

        AX2.set_title(DATA['score_title'])
        AX2.set_xlabel('Training Steps')
        AX2.set_ylabel('Score')

        for i in range(score_Y.shape[0]):
            AX2.plot(x, score_Y[i].flatten(), label=DATA['score_labels'][i])
        AX2.legend()

        AX3.set_title('Spectra')
        AX3.set_xlabel('Frequency')
        AX3.set_ylabel('Power (dB)')

        example = DATA['example']

        spectra = np.absolute(fft_data([DATA['example']])[0]).real
        m = np.amax(spectra)
        spectra = 20 * np.log10(spectra / m)
        AX3.plot(freq, spectra)

        AX4.set_title('Example')
        AX4.set_xlabel('Time')
        AX4.set_ylabel('Axis')

        AX4.plot(example.real)

    return
