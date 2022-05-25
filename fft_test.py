import re
import numpy as np
import matplotlib.pyplot as plt
import scipy.fft
from scipy.io import wavfile

SAMPLE_RATE = 192000

def read_file(filename: str, start_time: int, end_time: int) -> tuple[np.ndarray, int]:
    # Open the file and convert to mono
    sr, data = wavfile.read(filename)
    if data.ndim > 1:
        data = data[:, 0]

    # Return a slice of the data from start_time to end_time
    return (data[int(start_time * sr / 1000) : int(end_time * sr / 1000) + 1], sr)

def sine(freq: int, sr: int) -> np.ndarray:
    return np.sin(freq * np.arange(sr))

def draw(data: np.ndarray):
    plt.plot(data)
    plt.show()

def calc_fft(data: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(data)
    A = scipy.fft.fft(data)
    f = scipy.fft.fftfreq(n, 1 / sr)
    return (A, f)
    
duration = 1000
intervalls = 10
step_size = duration / intervalls
sr = SAMPLE_RATE
data = sine(0.1 * 2 * np.pi, sr)

start = 0

for i in range(0, duration, int(step_size)):

    tmp = data[i : int(i + step_size + 1)]

    a, f = calc_fft(tmp, sr)

    plt.scatter(f, np.abs(a), s=1)
    plt.show()


s = sine(0.1, SAMPLE_RATE)

draw(s)

a, f = calc_fft(s, SAMPLE_RATE)
plt.scatter(f, np.abs(a), s=0.1)
plt.show()

data, sr = read_file("C:\\Users\\Student\\Downloads\\white_noise.wav", 0, 1000)

print(sr)

a, f = calc_fft(data, sr)
plt.scatter(f, np.abs(a), s=1)
plt.show()




