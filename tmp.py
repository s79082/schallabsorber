from scipy.io.wavfile import read
import audiofile
import numpy as np
import matplotlib.pyplot as plt

from fft_util import calc_fft

SAMPLERATE = 48000

INTERVAL = (1115000, 1200000)
INTERVAL_LEN = INTERVAL[1] - INTERVAL[0]

# amount of frames we split the data
N_FRAMES = 50

data, sr = audiofile.read("data/wav1.wav")
data = data[INTERVAL[0] : INTERVAL[1]]
freqs, amps = calc_fft(data, sr)

db = 10 * np.log10(abs(amps) / 10 ** (-10.5))
print(max(db))

frame_size = int(SAMPLERATE / N_FRAMES)
    # split data into frames
frames = [ data[start : start + frame_size] for start in range(0, SAMPLERATE, frame_size) ]
print(len(frames))

# we assume the fist fram eholds the highest amp
build_max_matrix = True

for t, frame in enumerate(frames):
    freqs, amps = calc_fft(frame, sr)
    # TODO filter negative freqeusncies
    # TODO check for linearity  
#    freqs, amps = np.array(list(filter(lambda x: 2000s0.0 > x[0] >= 0, zip(freqs, amps))))
    print(len(freqs))
    if build_max_matrix:
        build_max_matrix = False

        maximas = np.ndarray((len(freqs),), dtype=np.float64)

        rts = np.zeros((len(freqs),), dtype=np.float64)
        last_diff = np.zeros((len(freqs),), dtype=np.float64)

        for idx, a in enumerate(amps): 
            maximas[idx] = 10 * np.log10(abs(a) / 10 ** (-10.5))

    db_amps = 10 * np.log10(abs(amps) / 10 ** (-10.5))

    diff = maximas - db_amps

    linearity = last_diff - diff 

    last_diff = diff

    mypos = 200
    #print(max(diff))

    for idx, (f, diff_db, lin) in enumerate(zip(freqs, diff, linearity)):
        if diff[idx] > 20.0 and diff[idx] != -1 and abs(lin) < 0.01:
            #print("put", t)
            print(lin)
            rts[idx] = t
            diff[idx] = -1


    #print(t)
print(rts)
print(min(rts))

rts *= frame_size
rts /= SAMPLERATE

plt.scatter(freqs, rts)
#plt.scatter(freqs, rts)
#plt.bar(x=freqs, height=rts, width=20)
plt.show()