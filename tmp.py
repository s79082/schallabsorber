from scipy.io.wavfile import read
import audiofile
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal.windows as win

from fft_util import calc_fft, generate_test_sine
from reverberation_detector import detect_intervals, interval_len
    
LINEAR_CRITERIUM = 5
X_0 = 3.3656 * 10 ** -11

DB_DIFF = 20.0 

def to_dB(data: np.ndarray) -> np.ndarray:
    return 10 * np.log10(abs(data) / X_0)

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

# read the data
#data, sr = audiofile.read("data/wav1.wav")

# cut away a part of interest
#data = data[INTERVAL[0] : INTERVAL[1]]

def calc_rt(data: np.array) -> np.array:

    global nfreqs
    # split data into frames
    frame_size = int(SAMPLERATE / N_FRAMES)
    frames = [ data[start : start + frame_size] for start in range(0, SAMPLERATE, frame_size) ]

    windowed_frames = [ win.hann(len(frame)) * frame for frame in frames ]
    #windowed_frames = frames
    # calc the fft for each frame
    spectrum_each_frame = [ calc_fft(frame, SAMPLERATE) for frame in windowed_frames ]

    # we assume the fist frame holds the highest amp
    build_max_matrix = True

    maximas = None

    freq_collection = dict()


    for t, (freqs, amps) in enumerate(spectrum_each_frame):


        # TODO filter negative freqeusncies
        #freqs, amps = np.array(list(filter(lambda x: 20000.0 > x[0] >= 0, zip(freqs, amps))))
        if build_max_matrix:
            build_max_matrix = False

            # maximal amplitude for each freq in db
            #maximas = np.ndarray((len(freqs),), dtype=np.float64)
            for f in freqs:
                freq_collection[f] = tuple()
            rts = np.zeros((len(freqs),), dtype=np.float64)

            # diff drom last frame
            last_diff = np.zeros((len(freqs),), dtype=np.float64)

            # populate maximas; assume first frame contains the maximas
            maximas = to_dB(amps)

            finished = np.ones((len(freqs), ), dtype=np.int8)

        # convert to dB
        db_amps = to_dB(amps)

        # calc diff
        diff = maximas - db_amps

        # calculate the difference between differences
        linearity = last_diff - diff 

        # finally, prepare for next frame
        last_diff = diff

        # print("frame nr " , str(t))
        # print("freq", freqs[92])
        # print("amps", db_amps[92])
        # print("diff", diff[92])
        # print("linear", linearity[92])


        # check if RT can be calculated
        for idx, (f, diff_db, lin, fin) in enumerate(zip(freqs, diff, linearity, finished)):
            if diff_db > DB_DIFF and fin == 1:
                finished[idx] = -1
                print("diff over 20", f)
                if abs(lin) < LINEAR_CRITERIUM:
        
                    #print("################ !!!!! ############")
                    rts[idx] = t
                


        #print(t)
    print(rts)
    print(min(rts))

    width = freqs[1] - freqs[0]

    # calc actual rt in seconds
    rts *= frame_size
    rts /= SAMPLERATE

    max_rt = np.max(rts)
    res = np.where(rts == max_rt)
    print(freqs[res[0]])
    print(rts[res[0]])


    # draw rt
    #plt.scatter(freqs, rts)
    #plt.scatter(freqs, rts)
    plt.title("Nachhallzeit")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("RT20 (s)")

    plt.bar(x=freqs, height=rts, width=width, color="orange")
    plt.show()
    print("nfreqs", len(freqs))
    return rts

if __name__ == "__main__":
    # load file
    data, sr = audiofile.read("data/wav1.wav")
    
    # detect interests
    interval = detect_intervals(data)

    matrices = []

    #rt_acc = np.zeros((480,), dtype=np.float64)

    import scipy.signal as sig

    acc_mat = None
    init_acc = True
    n_mat = 0

    # for each interest
    for _, end in interval:

        #end += 30000
        data_slice = data[end - 16000 : end]
        print(len(data_slice))
        

        #cutoff = len(data_slice) - 50000

        f, t, M = sig.spectrogram(data_slice, nperseg=50)
        #plt.pcolormesh(t, f, M)
        #plt.show()

        index_0_04 = np.where(f == 0.3)[0][0]

        tmp = M[index_0_04]

        plt.scatter(t, tmp)
        plt.show()

        print(index_0_04)

        print(M.shape)
        print(len(f))
        print(len(t))

        if init_acc:
            acc_mat = np.zeros(M.shape)
            init_acc = False
        acc_mat += M
        n_mat += 1

        #data_slice = data_slice[cutoff:]

        #plt.plot(data)
        #plt.vlines([start, end], 0, max(data), "red")
        #plt.vlines([start + cutoff], 0, max(data), "green")

        #rt_acc += calc_rt(data_slice)


    # calculate average
    acc_mat /=4

    plt.pcolormesh(t, f, acc_mat)
    plt.show()

    M_db = 10 * np.log10(abs(acc_mat) / X_0)

    plt.pcolormesh(t, f, M_db)
    plt.show()

    # zeros evaluate to false
    rt_calculated = np.zeros(f.shape, dtype=bool)

    # samples for each freq
    for f_idx, samples_over_time in enumerate(acc_mat):
        dbs = to_dB(samples_over_time)
        maxima = dbs[0]
        # cut away start
        dbs = dbs[1:]
        for t_idx, sample in enumerate(dbs):
            if maxima - sample > DB_DIFF and rt_calculated[f_idx] is False:
                print(f[f_idx], "found")
                rt_calculated[f_idx] = True


print(f)
    #rt_acc /= 4

    #plt.plot(rt_acc)
    #plt.show()







    
