import re
from wave import Wave_read
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

from scipy.fft import fft, fftfreq

def calc_fft(data: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """
        calculates the fft for a given array of data and a samplerate sr.
        returns (frequency, amplitude)

    """
    #X = fft(data, norm="forward")
    X = fft(data)
    N = len(X)
    freq = fftfreq(N, 1 / sr)
    return (freq[:int(N/2)], np.abs(X[:int(N/2)]))

def generate_test_sine(l: int):

    # gedämpfte 
    nums = np.arange(l)
    sine1 = np.zeros(l)
    sine2 = np.zeros(l)
    sine3 = np.zeros(l)
    sine4 = np.zeros(l)
    sine6 = np.zeros(l)
    sine7 = np.zeros(l)

    sine1 = np.power(10, -0.0000025 * nums) *  np.sin(0.01 * 2 * np.pi * nums)
    sine2 = np.power(10, -0.000005 * nums) *  np.sin(0.005 * 2 * np.pi * nums)
    sine3 = np.power(10, -0.000005 * nums) *  np.sin(0.0025 * 2 * np.pi * nums)
    sine4 = np.power(10, -0.0000025 * nums) *  np.sin(0.02 * 2 * np.pi * nums)

    sine5 = np.power(10, -0.000006 * nums) *  np.sin(0.025 * 2 * np.pi * nums)

    sine6 = np.power(10, -0.000004 * nums) *  np.sin(0.0225 * 2 * np.pi * nums)

    sine7 = np.power(10, -0.0000025 * nums) *  np.sin(0.003 * 2 * np.pi * nums)

    final_sine = sine1 + sine2 + sine3 + sine4 + sine5 + sine6 + sine7

    plt.plot(sine1)
    plt.plot(sine2)
    plt.plot(sine3)
    plt.plot(sine4)
    plt.plot(sine5)
    plt.plot(sine6)
    plt.plot(sine7)
    plt.title("einzelne Frequenzen")
    plt.show()

    plt.plot(final_sine)
    plt.title("kombinierte Frequenzen")
    plt.show()

    return final_sine

def generate_noise(l: int) -> np.ndarray:
    
    nums = np.arange(l)
    final_sine = np.zeros(l)

    for i in range(10):
        d = -(0.0000025 + i * 0.0000001)
        f = 0.02 + i * 0.001
        final_sine += np.power(10, d * nums) *  np.sin(f * 2 * np.pi * nums)

    plt.plot(final_sine)
    plt.show()
    return final_sine

def same_rt_diff_freq(l: int) -> np.ndarray:
    nums = np.arange(l)
    final_sine = np.zeros(l)

    d = -0.000025

    for i in range(20):
        
        f = 0.02 + i * 0.001
        final_sine += np.power(10, d * nums) *  np.sin(f * 2 * np.pi * nums)

    return final_sine

def to_db(data: np.ndarray) -> list:
    return [ 20 * np.log10(abs(chunk)) for chunk in data]

def read_file_raw(fname: str):
    import wave
    f: Wave_read = wave.open(fname)

    ret: np.ndarray = np.array(0)
    for byte_nr in range(1, f.getnframes()):
        byte = f.readframes(byte_nr)

        val: int = int.from_bytes(byte, "big")

        ret = np.append(ret, [val])

    return ret
    
def translate(leftMin, leftMax, rightMin, rightMax):

    def t(value):
        # Figure out how 'wide' each range is
        leftSpan = leftMax - leftMin
        rightSpan = rightMax - rightMin

        # Convert the left range into a 0-1 range (float)
        valueScaled = float(value - leftMin) / float(leftSpan)

        # Convert the 0-1 range into a value in the right range.
        return rightMin + (valueScaled * rightSpan)

    return np.vectorize(t)

max_amp = lambda f,a : max(zip(f,a), key=lambda x: x[1])
