import re
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

def read_file(filename: str, start_time: int, end_time: int) -> tuple[np.ndarray, int]:
    # Open the file and convert to mono
    sr, data = wavfile.read(filename)
    if data.ndim > 1:
        data = data[:, 0]

    # Return a slice of the data from start_time to end_time
    return (data[int(start_time * sr / 1000) : int(end_time * sr / 1000)], sr)

plt.style.use('seaborn-poster')

# sampling rate
sr = 192000
# sampling interval
ts = 1.0/sr
t = np.arange(0,1,ts)

freq = 1.
x = 3*np.sin(2*np.pi*freq*t)

freq = 4
x += np.sin(2*np.pi*freq*t)

freq = 7   
x += 0.5* np.sin(2*np.pi*freq*t)

x, sr = read_file("C:\\Users\\Student\\Downloads\\white_noise.wav", 0, 1000)

ts = 1.0/sr
t = np.arange(0,1,ts)

# plt.figure(figsize = (8, 6))
# plt.scatter(t, x)
# plt.ylabel('Amplitude')

# plt.show()



from scipy.fft import fft, fftfreq

def calc_fft(data: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """
        calculates the fft for a given array of data and a samplerate sr.
        returns (frequency, amplitude)

    """
    #print(len(data))
    X = fft(data)
    #print(X)
    N = len(X)
    n = np.arange(N)
    T = N/sr
    #freq = n/T 
    freq = fftfreq(N, 1 / sr)
    #print(len(freq))
    return (freq, np.abs(X))
    #return (freq, np.real(X))
