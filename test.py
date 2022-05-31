import numpy as np
import matplotlib.pyplot as plt
from fft_util import calc_fft as fft

SAMPLERATE = 192000

def generate_test_sine():

    # gedämpfte 
    nums = np.arange(SAMPLERATE)
    sine1 = np.zeros(SAMPLERATE)
    sine2 = np.zeros(SAMPLERATE)
    sine1 = np.power(10, -0.0000025 * nums) *  np.sin(0.01 * 2 * np.pi * nums)
    #sine2 = np.power(10, -0.000045 * nums) *  np.sin(0.005 * 2 * np.pi * nums)
    sine2 = np.power(10, -0.000005 * nums) *  np.sin(0.005 * 2 * np.pi * nums)

    final_sine = sine1 + sine2

    #final_sine = np.sin(2 * np.pi * nums)
    #plt.plot(final_sine)
    plt.plot(sine1)
    plt.plot(sine2)
    plt.show()
    return final_sine

def find(data: np.ndarray, n: int = 1) -> list:
    
    pass

data = generate_test_sine()

# show whole fft
fs, amps = fft(data, SAMPLERATE)

plt.scatter(fs, amps)
plt.show()


# split data
frame_size = int(SAMPLERATE / 40)
frames = [ data[start : start + frame_size] for start in range(0, SAMPLERATE, frame_size) ]

# calc fft for each frame
fft_data = [ fft(frame, SAMPLERATE) for frame in frames ]

# measure rt for each frequency
n_freqs = len(list(filter(lambda x: x >= 0, fft_data[0][0])))
#max_freqs = np.zeros(n_freqs)
#wanted_diff = np.zeros(n_freqs)
max_amp = dict()
#wanted_diff = dict()
wanted_diff = max(fft_data[0][1]) * 0.4
rt_for_freq = dict()


for t, (f, a) in enumerate(fft_data):

    # cut away negative frequencies
    tmp_zip = np.array(list(filter(lambda x: x[0] >= 0, zip(f, a))))

    for i, (freq, amp) in enumerate(tmp_zip):

        if freq == 75000.0:
        #if freq == 1920 or freq == 960:
            print(freq, amp)

        # RT has already been calculated for this frequency
        if max_amp.get(freq) == -1:
            continue

        #if max_freqs[i] == 0:
        if not freq in max_amp:
            # first frame, set amp, assuming its the max
            #max_freqs[i] = amp
            #wanted_diff[i] = 0.4 * amp
            max_amp[freq] = amp
            #wanted_diff[freq] = 0.4 * amp

            continue

        #if max_freqs[i] - amp >= wanted_diff[i]:
        if max_amp[freq] - amp >= wanted_diff:

            # mark freq as done
            max_amp[freq] = -1

            samples = t * frame_size

            rt_for_freq[freq] = samples / SAMPLERATE
            #if samples > 19200:
                #print("RT for freq {} is {} ({} samples)".format(freq, str(samples / SAMPLERATE), samples))

for key, val in rt_for_freq.items():
    if val is not None:
        print(key, val)
print(rt_for_freq[1920.0])
print(rt_for_freq[960.0])
print(rt_for_freq.get(1910.0))

#     sorted_index_array = np.argsort(np.array(list(map(lambda x: x[1], tmp_zip))))
  
#     # sorted array
#     sorted_array = tmp_zip[sorted_index_array]
#     rslt = sorted_array[-2:]
#     print(rslt)
#     plt.scatter(f, a)
#     plt.show()

# max_freq_each_split = [ max(zip(fft_res[0], fft_res[1]), key=lambda x: x[1]) for fft_res in fft_data ] 

# print(max_freq_each_split)
