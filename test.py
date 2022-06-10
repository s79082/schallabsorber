from signal import signal
import colored
from matplotlib import colors
import numpy as np
import matplotlib.pyplot as plt
from fft_util import calc_fft, generate_test_sine, generate_noise, same_rt_diff_freq
import peakutils

import scipy.signal.windows as win

from scipy.io import wavfile

SAMPLERATE = 48000

INTERVAL = (1000000, 1200000)

# amount of frames we split the data
N_FRAMES = 10

def main():
    

    # load datd
    #data = generate_test_sine(SAMPLERATE)
    #data = generate_noise(SAMPLERATE)
    #data = same_rt_diff_freq(SAMPLERATE)
    sr, data = wavfile.read("data/wav1.wav")
    # cut out 
    data = data[INTERVAL[0] : INTERVAL[1]]

    # show spektrum for whole data
    fs, amps = calc_fft(data, SAMPLERATE)

    peak_indexes = peakutils.indexes(amps, thres=0.5)
    print("peaks")
    print(len(amps[peak_indexes]))

    y_min = np.zeros(len(peak_indexes))
    y_max = np.array(amps[peak_indexes])

    #plt.vlines(peak_indexes, ymin=y_min, ymax=y_max, colors=[(255, 0, 0, 125)])
    plt.scatter(fs, amps)
    plt.title("spectrum all samples")
    plt.show()

    frame_size = int(SAMPLERATE / N_FRAMES)
    # split data into frames
    frames = [ data[start : start + frame_size] for start in range(0, SAMPLERATE, frame_size) ]

    plt.plot(data * win.hann(len(data)))
    plt.show()

    # window frames with hanning window
    windowed_frames = [ win.hann(len(frame)) * frame for frame in frames ]

    #plt.plot(frames[0] * win.hann(len(frames[0])))
    #plt.show()
    #for spec_frame in windowed_frames:
    #    plt.plot(spec_frame)
    #    plt.show()

    # calc fft for each frame
    spectrum_each_frame = [ calc_fft(frame, SAMPLERATE) for frame in windowed_frames ]

    
    for spec_frame in spectrum_each_frame:
         #plt.bar(spec_frame[0], spec_frame[1], width=20)
        plt.scatter(spec_frame[0], spec_frame[1])
        plt.show()


    # the max map for each freq
    max_amp = dict()

    # the wanted difference in amplitude; TODO this should relate to 60dB for RT60
    # now just take 40% of the max amp of the first frame
    wanted_diff = max(spectrum_each_frame[0][1]) * 0.5
    #wanted_diff = 5000
    print(wanted_diff)

    # RT in samples for each freq
    rt_for_freq = dict()

    for t, (f, a) in enumerate(spectrum_each_frame):

        # cut away negative frequencies and above 20kHz
        tmp_zip = np.array(list(filter(lambda x: 20000.0 > x[0] >= 0, zip(f, a))))

        for freq, amp in tmp_zip:

            # RT has already been calculated for this frequency
            if max_amp.get(freq) == -1:
                continue

            # max amp has not been set for this freq yet
            if not freq in max_amp:

                # first frame, set amp, assuming its the max
                max_amp[freq] = amp
                continue

            rt_for_freq[freq] = 0

            # the wanted diff (60dB) has been reached
            if max_amp[freq] - amp >= wanted_diff:            

                # mark freq as done
                # TODO do this smoother
                max_amp[freq] = -1

                # RT in samples
                samples = t * frame_size

                #rt_for_freq[freq] = samples / SAMPLERATE
                rt_for_freq[freq] = samples

                #print(freq, "found new diff")
                #print(freq, samples)

        # plot amps for each frame
        # plt.scatter(f, a)
        # plt.ylim(0, 10000)
        # plt.show()


    # print rt maxima
    maxima_freq = max(rt_for_freq, key=rt_for_freq.get)
    print(maxima_freq, rt_for_freq[maxima_freq])

    # display rt
    xaxis = np.array(list(rt_for_freq.keys()))
    yaxis = np.array(list(rt_for_freq.values()))

    print(yaxis)

    plt.bar(xaxis, yaxis, width=20)
    plt.ylabel("RT60 (samples)")
    plt.xlabel("Frequenz (Hz)")
    plt.title("Nachhallzeit für 192k samples")
    plt.show()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit()