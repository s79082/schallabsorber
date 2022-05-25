from numpy import mean
import numpy as np
from fft_util import calc_fft, read_file
#from fft_test import SAMPLE_RATE
import matplotlib.pyplot as plt

SAMPLERATE = 192000

def generate_test_sine():

    # gedämpfte 
    nums = np.arange(192000)
    #data = np.sin(0.001 * nums)
    data = np.power(10, -0.0000025 * nums) *  np.sin(0.01 * nums)
    plt.plot(data)
    plt.show()
    return data

def gen_intervals(start: int, end: int, step: int):
    """
        Generator that returns (start, end) of an interval
    """
    for f in range(start, end, step):
        yield f, f + step


# load data from file
#data, sr = read_file("C:\\Users\\Student\\Downloads\\white_noise.wav", 0, 1000)
generate_test_sine()
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

print("data ", len(data))
print("data slices", len(data_slices))

# calc fft for each of these intervals
spectrums_for_each_time = [ calc_fft(data_slice, sr) for data_slice in data_slices ]

freq_interval = (50.0, 100.0)

max_amp = None
tmp_amp = 0
wanted = 0

# get mean spektrum for freq 50 - 100Hz for each time frame
#for t in sample_intervals:
 #   avg = mean()
for t, spec in enumerate(spectrums_for_each_time):
    print("n freqs", len(spec[0]))
    print(spec[0])
    print("n amps", len(spec[1]))
    # filter negative freqencies and their amplitudes, should be half the amount
    positives = list(filter(lambda x: x[0] >= 0, zip(spec[0], spec[1])))
    #print(positives)
    #for f in zip(spec[0], spec[1]):
        #if f >= 0:
            #positives += f

    print("n positves", len(positives))

    freqs_in_interval = list(filter(lambda x: 50.0 < x[0] < 10000.0, positives))


    fs = list(map(lambda x: x[0], freqs_in_interval))
    amps = list(map(lambda x: x[1], freqs_in_interval))

    

    # cut spectrum into frequncy intervals
    freq_intervals = gen_intervals(100, 10000, 100)
#    for start, end in freq_intervals:
 #       spec_slice = list(filter(lambda x: start <= x[0] <= end, freqs_in_interval))
  #      slice_max = max(spec_slice, key=lambda x: x[1])
        #print(slice_max)

    avg = np.mean(amps)
    print(avg)

    slice_max = max(freqs_in_interval, key=lambda x: x[1])
    print(slice_max)
    tmp_amp = slice_max[1]

    if max_amp is None:
        max_amp = tmp_amp = slice_max[1]
        wanted = 0.6 * max_amp

    if max_amp - tmp_amp >= wanted:
        print("RT60: {} {} samples".format(str((t*step_size)/ SAMPLERATE), t*step_size))
        break

    #plt.scatter(fs, amps)
    #plt.show()

    # l_pos = list(list(positives)[0])

    # #print("pos", l_pos)
    # for p in l_pos:
    #     print("p", p)
    #     if p > 20.0:
    #         print(p)
    #         break
    # #print(list(positives)[0])
    # # filter frw in our interval
    # positives = list(filter(lambda x: freq_interval[0] <= x <= freq_interval[1], l_pos))
    # print("pos", len(positives))


    # amplitudes = map(lambda x: x[1], positives)
    
    # print("amps", len(list(amplitudes)))


    # amplitudes = filter(lambda x: x > 0, amplitudes)

    # print("amps", len(list(amplitudes)))

    # # find the avg amplitude
    # avg = mean(np.array(list(amplitudes)), dtype=np.float16)
    # print(avg)


print(len(spectrums_for_each_time[0][0]))
#print(spectrums_for_each_time[0])

freq_intervalls = range(0, 5000, freq_inter_size)

# iterate over freq intervals first
""" for freq in freq_intervalls:

    # current freq slice
    freq_slice = spectrums_for_each_time[freq, freq + freq_inter_size]

    for t in time_intervalls: """


#for spec in spectrums_for_each_time:

 #   for f1, f2 in gen_intervals(0, 5000, 100):

        
        

# slice freq
# for f1, f2 in gen_intervals(0, 5000, 100):

#     for t1, t2 in gen_intervals(0, 1000, 100):

#         time_slice = spectrums_for_each_time[ t1:t2 ]

        





#fin = []

time_idx = 0
freq_idx = 0

for spec in spectrums_for_each_time:

    spectrums_over_time = []

    # slice spectrum in frequency intervals
    for interval_slice in freq_intervalls:

        freq_slice = spec[interval_slice : interval_slice + freq_inter_size]

        # put this spectrum slice over time
        spectrums_over_time += freq_slice

    #freq_slices = [ spec[f : f + freq_inter_size] for f in freq_intervalls ]
    







# for i in range(0, duration, int(step_size)):

#     tmp = data[i : int(i + step_size)]

#     f, a= calc_fft(tmp, sr)

#     plt.scatter(range(int(step_size)), tmp, s=1)
#     plt.show()
#     print("here")

#     plt.scatter(f, a, s=1)
#     plt.show()
