from tracemalloc import stop
from numpy import arange, mean
import numpy as np
from fft_util import calc_fft, read_file
#from fft_test import SAMPLE_RATE
import matplotlib.pyplot as plt

SAMPLERATE = 192000

def generate_test_sine():

    # gedämpfte 
    nums = np.arange(SAMPLERATE)
    sine1 = np.zeros(SAMPLERATE)
    #sine1 = np.power(10, -0.0000025 * nums) *  np.sin(0.01 * 2 * np.pi * nums)
    sine2 = np.power(10, -0.000045 * nums) *  np.sin(0.005 * 2 * np.pi * nums)
    #final_sine = sine1 + sine2
    final_sine = np.sin(2 * np.pi * nums)
    plt.plot(final_sine)
    plt.show()
    return final_sine

def gen_intervals(start: int, end: int, step: int):
    """
        Generator that returns (start, end) of an interval
    """
    for f in range(start, end, step):
        yield f, f + step


# load data from file
#data, sr = read_file("C:\\Users\\Student\\Downloads\\white_noise.wav", 0, 1000)
nums = np.arange(192000 / 2)
first_part = np.sin(0.01 * nums)
second_part = np.sin(0.1 * nums)
#data, sr = np.concatenate([first_part, second_part]), 192000
data, sr = generate_test_sine(), SAMPLERATE
duration = 1000
n_samples = sr * 1000 / duration
intervalls = 100
step_size = n_samples / intervalls
start = 0

print("n_sanmples", n_samples)
print("step_size", step_size)

freq_inter_size = 100

f, a = calc_fft(data, sr)
#plt.scatter(f, a, s=1)
#plt.show()

sample_intervals = range(0, int(n_samples), int(step_size))

# slice the file in time intervals
data_slices = [ data[i : int(i + step_size)] for i in sample_intervals ]

print("len data ", len(data))
print("len data_slices", len(data_slices))

# calc fft for each of these intervals
spectrums_for_each_time = [ calc_fft(data_slice, sr) for data_slice in data_slices ]

freq_interval = (50.0, 100.0)


# highest amplitude
#max_amp = max(data)
max_amp = None
# amp each iteration 
tmp_amp = 0
wanted_amp = 0

acc_rts = []

for start, end in gen_intervals(50, 10000, 100):

    for t, spec in enumerate(spectrums_for_each_time):
        #print("n freqs", len(spec[0]))
        #print(spec[0])
        #print("n amps", len(spec[1]))
        # filter negative freqencies and their amplitudes, should be half the amount
        positives = list(filter(lambda x: x[0] >= 0, zip(spec[0], spec[1])))
        #print(positives)
        #for f in zip(spec[0], spec[1]):
            #if f >= 0:
                #positives += f

        #print("n positves", len(positives))
        

        #print(start, end)
        # filter for freq interval
        freqs_in_interval = list(filter(lambda x: start < x[0] < end, positives))

        # for plotting
        fs = list(map(lambda x: x[0], freqs_in_interval))
        amps = list(map(lambda x: x[1], freqs_in_interval))
        


        # cut spectrum into frequncy intervals
        freq_intervals = gen_intervals(100, 10000, 100)
    #    for start, end in freq_intervals:
    #       spec_slice = list(filter(lambda x: start <= x[0] <= end, freqs_in_interval))
    #      slice_max = max(spec_slice, key=lambda x: x[1])
            #print(slice_max)

        # get the point with the highest amp
        slice_max = max(freqs_in_interval, key=lambda x: x[1])
        #print(slice_max)
        tmp_amp = slice_max[1]

        if max_amp is None:
            max_amp = tmp_amp = slice_max[1]
            wanted_amp = 0.6 * max_amp

            #print("max_amp", max_amp)
            #print("wanted_amp", wanted_amp)

        if max_amp - tmp_amp >= wanted_amp:
            print("RT60: {} {} samples".format(str((t*step_size)/ SAMPLERATE), t*step_size))

            acc_rts.append((start, end, (t*step_size)/ SAMPLERATE))
            print("added", (start, end, (t*step_size)/ SAMPLERATE))

            # plt.title("data slice {}/{}".format(t, len(spectrums_for_each_time)))
            # plt.plot(data_slices[t])
            # plt.show()

            # plt.scatter(fs, amps)
            # plt.title("frquencies in interval [{},{}] at data slice {}".format(start, end, t + 1))
            # plt.show()

            # reset difference calculation
            max_amp = None
            tmp_amp = 0
            wanted_amp = 0
            break

print(max(acc_rts, key=lambda x: x[2]))
rt_plot = list(map(lambda x: (x[1] - x[0], x[2]), acc_rts))
fs = list(map(lambda x: x[0], rt_plot))
rt60s = list(map(lambda x: x[1], rt_plot))

plt.scatter(rt60s, fs)
plt.show()