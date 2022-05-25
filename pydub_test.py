import numpy as np
import matplotlib.pyplot as plt
import scipy.fft
from scipy.io import wavfile

import peakutils

from fft_util import calc_fft

def find_freq_for_max_amp(fs: list, amps: list):
    assert len(fs) == len(amps)
    _max = amps[0]
    _max_freq = 0
    for freq, amp in zip(fs, amps):
        if amp > _max:
            _max = amp
            _max_freq = freq
    
    return _max_freq



def find_n_max():
    pass

def intervalls(data: np.ndarray):
    start = 0
    for i in range(19):
        yield data[start : start + 100 + 1]
        start += 100

def read_file(filename: str, start_time: int, end_time: int):

    # Open the file and convert to mono
    sr, data = wavfile.read(filename)
    if data.ndim > 1:
        data = data[:, 0]

    # Return a slice of the data from start_time to end_time
    return (data[int(start_time * sr / 1000) : int(end_time * sr / 1000) + 1], sr)

def freq(data, sr, n=5):

    # Fourier Transform
    N = len(data)
    yf = scipy.fft.rfft(data)
    xf = scipy.fft.rfftfreq(N, 1 / sr)

    # Uncomment these to see the frequency spectrum as a plot
   

    # Get the most dominant frequency and return it


    """ for i in range(n):
        idx = np.argmax(yf)
        max_vals.append(xf[idx])
        yf[idx] = - np.Infinity """

    idx = np.argmax(yf)
    print("idx ", idx)
    freq = xf[idx]

    indexes_peaks = peakutils.indexes(yf, min_dist=1000, thres=100000, thres_abs=True)

    print(xf[indexes_peaks])

    #s = np.sort_complex(yf)
    #print(s[-n : ])
    sorted_idx = np.argsort(yf)

    sorted_freq = xf[sorted_idx]

    sorted_freq = list(sorted_freq[-n : ])
    sorted_freq.reverse()

    print("{} most dominant frequencies: ".format(n))
    for f in sorted_freq:
        print("{} Hz".format(f))

    #for i in sorted_freq:
     #   print(np.abs(i))

    #print(sorted_freq)

    #absolute = np.array(map(np.abs, yf))
    #print(sorted(absolute))
    #sorted_idx = np.argsort(map(np.abs, yf))

    plt.vlines(xf[indexes_peaks], 0, 30000000, colors="red")
    #plt.vlines(sorted_freq, 0, 50000000, colors="green")

    plt.plot(xf, np.abs(yf))
    plt.show()

    return freq


combined_sine = np.arange(192000)
combined_sine = np.sin(0.01 * combined_sine)
print(combined_sine.ndim)
#combined_sine = np.sin(combined_sine) + 3 * np.sin(0.5 * combined_sine) + 0.5 * np.sin(combined_sine)

#data, sr = read_file("C:\\Users\\Student\\Downloads\\sine.wav", 0, 1)


data, sr = read_file("C:\\Users\\Student\\Downloads\\sine.wav", 0, 1000)

plt.plot(data)
plt.show()

freqs, vals = calc_fft(data, sr)

plt.plot(freqs, vals)
plt.show()

#combined_sine = np.sin(combined_sine)
#sr = 200000
data, sr = combined_sine, 192000
print(sr)
#data , sr = combined_sine, 2000
start: int = 0
timestamp = 0
audio_lenght = 1000
time_interval_n = 5
time = range(time_interval_n)
time_interval_len: int = audio_lenght / time_interval_n 





for i in time:

    tmp = data[start : int(start + time_interval_len + 1)]
    N = len(tmp)
    yf = scipy.fft.rfft(tmp)
    xf = scipy.fft.rfftfreq(N, 1 / sr)

    _max_amp = yf[0]
    _max_freq = xf[0]

    """ for amp, _freq in zip(yf, xf):
        print(abs(np.real(amp)), _freq)
        if abs(np.real(amp)) > abs(_max_amp):
            print(abs(np.real(amp)), _max_amp)
            _max_amp = amp
            _max_freq = _freq """

    print(_max_freq)

    plt.plot(tmp)
    plt.show()

    print(find_freq_for_max_amp(xf, yf))

    idx = np.argmax(yf)
    max_freq = xf[idx]
    print(max_freq)

    print(len(yf))
    sorted_idx = np.argsort(yf)

    sorted_freq = xf[sorted_idx]

    sorted_freq = list(sorted_freq[-3 : ])
    sorted_freq.reverse()
    print(sorted_freq)

    plt.scatter(xf, np.abs(yf))
    plt.show()

    start += 200
    timestamp += 1

#plt.plot(combined_sine)
#plt.show()

#print(freq(combined_sine, 50000, n=3))
data, sr = read_file("C:\\Users\\Student\\Downloads\\sine.wav", 0, 2000)
plt.plot(data)
plt.show()
print(freq(data, sr, 5))

