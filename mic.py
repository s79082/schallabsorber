#!/usr/bin/env python3
"""Plot the live microphone signal(s) with matplotlib.

Matplotlib and NumPy have to be installed.

"""
import argparse
from ast import arg
import queue
from statistics import mean
import sys

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd

from tmp import to_dB
import tmp


def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    '-l', '--list-devices', action='store_true',
    help='show list of audio devices and exit')
args, remaining = parser.parse_known_args()
if args.list_devices:
    print(sd.query_devices())
    parser.exit(0)
parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    parents=[parser])
parser.add_argument(
    'channels', type=int, default=[1], nargs='*', metavar='CHANNEL',
    help='input channels to plot (default: the first)')
parser.add_argument(
    '-d', '--device', type=int_or_str,
    help='input device (numeric ID or substring)')
parser.add_argument(
    '-w', '--window', type=float, default=200, metavar='DURATION',
    help='visible time slot (default: %(default)s ms)')
parser.add_argument(
    '-i', '--interval', type=float, default=30,
    help='minimum time between plot updates (default: %(default)s ms)')
parser.add_argument(
    '-b', '--blocksize', type=int, help='block size (in samples)')
parser.add_argument(
    '-r', '--samplerate', type=float, help='sampling rate of audio device')
parser.add_argument(
    '-n', '--downsample', type=int, default=10, metavar='N',
    help='display every Nth sample (default: %(default)s)')
args = parser.parse_args(remaining)
if any(c < 1 for c in args.channels):
    parser.error('argument CHANNEL: must be >= 1')
mapping = [c - 1 for c in args.channels]  # Channel numbers start with 1
q = queue.Queue()
print(args)
start = None
end = 0
current = 0
draw = False
n_frames = 0
def audio_callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    global start, end, draw, n_frames, record_data
    if status:
        print(status, file=sys.stderr)
    #print(len(indata[5]))
    data = indata[:, mapping]
    data_dB = to_dB(data)
    #print(int(len(data) / 1000))
    chunky_data = np.array_split(data, int(len(data) / 128))
    #print(len(chunky_data[0]))
    #print(all([ len(x) == 1136 for x in chunky_data ]))
    # TODO
    for i, s in enumerate([data_dB]):
        #print(len(s))
        val = np.max(s)
        if val > 85 and start is None:
            print("START")
            start = i
            n_frames = 0
            continue
        if start is not None and val < 70:
            print("END")
            #print(start, end)
            print(n_frames / 44100.0)
            n_frames = 0
            end = i
            start = None
            draw = True
            #print(len(record_data) / 44100)
        elif start is not None:
            shift = len(data)
            #print(shift)
            record_data = np.roll(record_data, -shift, axis=0)
            #print(record_data[-shift:, :])
            #print(data[0])
            record_data[-shift:, :] = data

    # Fancy indexing with mapping creates a (necessary!) copy:
    #if np.max(to_dB(indata[::args.downsample, mapping])) > 80:
    #    print(np.max(to_dB(indata[::args.downsample, mapping])))
    q.put(indata[::args.downsample, mapping])
    n_frames += frames
    

def update_plot(frame):
    """This is called by matplotlib for each plot update.

    Typically, audio callbacks happen more frequently than plot updates,
    therefore the queue tends to contain multiple blocks of audio data.

    """
    global plotdata, draw, ax, record_data
    while True:
        try:
            data = q.get_nowait()
        except queue.Empty:
            break
        shift = len(data)
        plotdata = np.roll(plotdata, -shift, axis=0)
        plotdata[-shift:, :] = data
    for column, line in enumerate(lines):
        if draw:
            #ax.vlines([start], [-10], [10], colors=["red"])
            #ax.vlines(end, -10, 10, colors=["red"])
            plt.ion()
            plt.figure().clear()
            #plt.plot(record_data)
            #plt.show()
            data_slice = np.array([ x for x in record_data if x != 0 ])


            

            #plt.plot(np.diff(split_maxs))
            #plt.show()

            #plt.plot(to_dB(data_slice))
            #plt.show()

            tmp.main(data_slice)
            record_data = np.zeros(record_data.shape)
            #ax[1].plot(record_data)
            draw = False
        else:
            line.set_ydata(plotdata[:, column])
    return lines

try:
    if args.samplerate is None:
        device_info = sd.query_devices(args.device, 'input')
        args.samplerate = device_info['default_samplerate']
    print(args.samplerate)
    length = int(args.window * args.samplerate / (1000 * args.downsample))
    plotdata = np.zeros((length, len(args.channels)))
    record_data = np.zeros((length * 100, len(args.channels)))
    print(length)

    fig, ax = plt.subplots(ncols=2)
    lines = ax[0].plot(plotdata)
    if len(args.channels) > 1:
        ax.legend([f'channel {c}' for c in args.channels],
                  loc='lower left', ncol=len(args.channels))
    ax[0].axis((0, len(plotdata), -1, 1))
    ax[0].set_yticks([0])
    ax[0].yaxis.grid(True)
    
    ax[0].tick_params(bottom=False, top=False, labelbottom=False,
                   right=False, left=False, labelleft=False)
    fig.tight_layout(pad=0)

    stream = sd.InputStream(
        device=None, channels=max(args.channels),
        samplerate=args.samplerate, callback=audio_callback)
    ani = FuncAnimation(fig, update_plot, interval=args.interval, blit=True)
    with stream:
        plt.show()
except Exception as e:
    parser.exit(type(e).__name__ + ': ' + str(e))