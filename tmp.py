from scipy.io.wavfile import read
import audiofile
import numpy as np
import matplotlib.pyplot as plt

from fft_util import calc_fft, generate_test_sine

LINEAR_CRITERIUM = 0.4
X_0 = -10.5

dB_20 = 20.0 

def to_dB(data: np.ndarray) -> np.ndarray:
    return 10 * np.log10(abs(data) / 10 ** X_0)

def filter_array(arr: np.array, fn) -> np.array:

    filter_list = []
    for el in arr:
        if fn(el):
            filter_list.append(True)
        else:
            filter_list.append(False)
    
    return arr[filter_list]

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

# split data into frames
frame_size = int(SAMPLERATE / N_FRAMES)
frames = [ data[start : start + frame_size] for start in range(0, SAMPLERATE, frame_size) ]


# we assume the fist frame holds the highest amp
build_max_matrix = True

maximas = None


for t, frame in enumerate(frames):


    freqs, amps = calc_fft(frame, sr)
    #tmp_zip = filter(lambda x: x[0] >= 0, zip(freqs, amps))
    #freqs = np.array(list(map(lambda x: x[0], tmp_zip)))
    #amps = np.array(list(map(lambda x: x[1], tmp_zip)))

    # TODO filter negative freqeusncies
    #freqs, amps = np.array(list(filter(lambda x: 20000.0 > x[0] >= 0, zip(freqs, amps))))
    if build_max_matrix:
        build_max_matrix = False

        # maximal amplitude for each freq in db
        #maximas = np.ndarray((len(freqs),), dtype=np.float64)


        rts = np.zeros((len(freqs),), dtype=np.float64)

        # diff drom last frame
        last_diff = np.zeros((len(freqs),), dtype=np.float64)

        # populate maximas; assume first frame contains the maximas
        maximas = to_dB(amps)

        finished = np.ones((len(freqs), ), dtype=np.int8)

    db_amps = to_dB(amps)

    # calc diff
    diff = maximas - db_amps

    # calculate the difference between differences
    linearity = last_diff - diff 

    # finally, prepare for next frame
    last_diff = diff

    # check if RT can be calculated
    for idx, (f, diff_db, lin, fin) in enumerate(zip(freqs, diff, linearity, finished)):
        if diff_db > 20.0 and fin == 1 and abs(lin) < LINEAR_CRITERIUM:
        #if diff[idx] > 20.0 and diff[idx] != -1 and abs(lin) < 0.01:
            #print("put", t)
            print(lin)
            rts[idx] = t
            finished[idx] = -1


    #print(t)
print(rts)
print(min(rts))

rts *= frame_size
rts /= SAMPLERATE

plt.scatter(freqs, rts)
#plt.scatter(freqs, rts)
#plt.bar(x=freqs, height=rts, width=20)
plt.show()