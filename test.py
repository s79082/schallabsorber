import numpy as np
import matplotlib.pyplot as plt
from fft_util import calc_fft as fft
import peakutils

SAMPLERATE = 192000

def generate_test_sine():

    # gedämpfte 
    nums = np.arange(SAMPLERATE)
    sine1 = np.zeros(SAMPLERATE)
    sine1 = np.power(10, -0.0000025 * nums) *  np.sin(0.01 * 2 * np.pi * nums)
    sine2 = np.power(10, -0.000045 * nums) *  np.sin(0.005 * 2 * np.pi * nums)
    final_sine = sine1 + sine2
    #final_sine = np.sin(2 * np.pi * nums)
    #plt.plot(final_sine)
    #plt.show()
    return final_sine

def find(data: np.ndarray, n: int = 1) -> list:
    
    pass

data = generate_test_sine()

# split data
step = int(SAMPLERATE / 10)
split_data = [ data[start : start + step] for start in range(0, SAMPLERATE, step)]

# calc fft for each frame
fft_data = [ fft(split, SAMPLERATE) for split in split_data ]

# measure rt for each frequency
freqs = np.zeros(len(list(filter(lambda x: x >= 0, fft_data[0][0]))))
print(len(freqs))

for f, a in fft_data:

    
    # TODO use peak detector
    #tmp_zip = np.array(list(zip(f, a)))

    # cut away negative frequencies
    tmp_zip = np.array(list(filter(lambda x: x[0] >= 0, zip(f, a))))

    sorted_index_array = np.argsort(np.array(list(map(lambda x: x[1], tmp_zip))))
  
    # sorted array
    sorted_array = tmp_zip[sorted_index_array]
    rslt = sorted_array[-2:]
    print(rslt)
    plt.scatter(f, a)
    plt.show()

max_freq_each_split = [ max(zip(fft_res[0], fft_res[1]), key=lambda x: x[1]) for fft_res in fft_data ] 

print(max_freq_each_split)
