
from typing import Callable
from pyparsing import col
from scipy.io.wavfile import read
import audiofile
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal.windows as win



from mpl_toolkits import mplot3d



from fft_util import calc_fft, generate_test_sine
from reverberation_detector import detect_intervals, interval_len

from sklearn.metrics import mean_squared_error

import tkinter as tk
from tkinter import END, Misc, ttk
    
LINEAR_CRITERIUM = 9.5
X_0 = 1.18111 * 10 ** -9
#X_0 = 3.36566 * 10 ** -11
#X_0 = 8.26072 * 10 ** -11


DB_DIFF = 60.0 

def to_dB(data: np.ndarray) -> np.ndarray:
    return 10 * np.log10(abs(data) / X_0)

SAMPLERATE = 48000

INTERVAL = (1115000, 1200000)
INTERVAL_LEN = INTERVAL[1] - INTERVAL[0]

# amount of frames we split the data
N_FRAMES = 50

all_rts = []


def calc():

    def f(data, nfft, nperseg, max_mse, max_slope, thresh):

        global axis, glob_rts, glob_freqs

        for a in axis:
            a.clear()

        print(nfft, nperseg)
        print(type(data))

        print(data.shape)
        tmp_data = np.zeros((data.shape[0],))
        for idx, _ in enumerate(tmp_data):
            tmp_data[idx] = data[idx]

        data = tmp_data

        print(type(data), len(data))
   
        interval = [(0, len(data))]
        matrices = []

        sr = 48000

        import scipy.signal as sig

        acc_mat = None
        init_acc = True
        n_mat = 0

        # split_size = 1000

        # # split data into slices
        # splits = np.array_split(to_dB(data), len(data) / split_size)

        # split_maxs = [max(split) for split in splits]

        # diffs = np.diff(split_maxs)

        # smaller_than = lambda c: lambda x: x < c

        # #idx_drop = idx_where(diffs, lambda x: x < -4)
        # idx_drop = idx_where(diffs, smaller_than(-2))
        # print("idx_drop", idx_drop)

        # if idx_drop > 5:
        #     # scale idx to real
        #     idx_drop *= split_size

        #     data = data[idx_drop:]

        # for each interest
        #data = data[-8000:]
        #data = data[-SAMPLERATE:]
        #import wave
        #with wave.open("test.wav", "wb") as wav:
        #    wav.setnchannels(1)
        #    wav.setsampwidth(1)
        #    wav.setparams()
        #    wav.writeframes(data)

        for _, end in interval:

            data_slice = data

            f, ts, M = sig.spectrogram(data_slice, fs=sr, window="hann", nfft=nfft, nperseg=nperseg)
            
            print(M.shape)
            print(len(f))
            print(len(ts))

            if init_acc:
                acc_mat = np.zeros(M.shape)
                init_acc = False
            acc_mat += M
            n_mat += 1


        # calculate average
        acc_mat /= len(interval)
 
        print("len M", len(M))
        M_tmp = np.zeros((M.shape[0], M.shape[1],), dtype=np.float32)
        for r, row in enumerate(M_tmp):
            for c, col in enumerate(row):
                M_tmp[r, c] = col

        print(type(M_tmp[0, 0]), M_tmp.shape)

        print(len(ts), len(f))
        axis[0].plot(to_dB(data_slice))
        axis[0].set_xlabel("time (s)")
        axis[0].set_ylabel("Pegel (dB)")

        M_db = to_dB(acc_mat)
        
        axis[1].pcolormesh(ts, f, M_db)
        axis[1].set_xlabel("time (s)")
        axis[1].set_ylabel("frequency (Hz)")
  
        # zeros evaluate to false
        rt_calculated = np.zeros(f.shape, dtype=bool)
        rts = np.zeros(f.shape)
        mses = np.zeros(f.shape)
        # contains the freq points where no rt could be calculated
        no_values = np.empty(f.shape, dtype=object)
        # contains all rts, not dep. on mses
        rts_no_mse = np.zeros(f.shape)
        slopes = np.zeros(f.shape)

        #rts = []

        # samples for each freq
        for f_idx, samples_over_time in enumerate(acc_mat):

            dBs = to_dB(samples_over_time)
            if 850 < f[f_idx] < 1150 or 4000 < f[f_idx] < 4500:

                print("max", max(dBs))

            if max(dBs) < thresh:
                rts[f_idx] = 0 
                continue

            # linear regression 
            model = np.polyfit(ts, dBs, 1)

            # get linear parameters
            slope = model[0]
            intersect = model[1]

            # amp we want 
            y_end = intersect - DB_DIFF

            # get predicted timestep
            t_predict = (y_end - intersect) / slope

            # calc mse for linear check
            predicted_dBs = slope * ts + intersect

            
            mse = mean_squared_error(dBs, predicted_dBs)

            mses[f_idx] = mse
            slopes[f_idx] = slope
            rts_no_mse[f_idx] = t_predict

            if mse <= max_mse and t_predict > 0:
                rts[f_idx] = t_predict  

            if slope > max_slope:
                rts[f_idx] = 0

        #f = f[:175]
        #print(f)
        #rts = rts[:175]
        all_rts.append(rts)
        print(rts[-1:])
        #axis[2].scatter(f, rts, s=5)
        axis[2].set_xlabel("frequency (Hz)")
        axis[2].set_ylabel("RT60 (s)")
        axis[2].scatter(f, mses, c=['red'], s=3)
        for messung, color in zip(all_rts, ['red', 'blue', 'green']):
            axis[2].scatter(f, messung, s=5, c=[color])

        

        axis[2].plot([250, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000],
            [0.98, 0.53, 0.82, 0.8, 0.6, 0.71, 0.69, 0.63, 0.68, 0.64, 0.63, 0.63, 0.63])

        #2
        axis[2].plot([400, 500, 800, 1000, 1250, 1600, 2000, 2500, 3150],
        [0.44, 0.7, 0.66, 0.72, 0.71, 0.64, 0.68, 0.64, 0.63])

        #0
        axis[2].plot([500, 800, 1000, 1250, 1600, 2000, 2500, 3150],
        [0.81, 0.73, 0.7, 0.77, 0.68, 0.7, 0.69, 0.67])
        #3
        axis[2].plot([250, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150],
            [0.5, 0.83, 0.68, 0.7, 0.73, 0.62, 0.68, 0.65, 0.61])
        #axis[3].plot(diffs)
        #axis[3].fig = plt.figure()
        #ax = plt.axes(projection='3d')
        

        print(len(f), f[1] - f[0])

        glob_rts = rts
        glob_freqs = f
        write_file()

        plt.show()


    return f

def idx_where(arr: np.ndarray, key) -> int:

    for i, a in enumerate(arr):
        if key(a):
            return i
    
    return -1


def write_file():
    #global glob_freqs, glob_rts, txt_filename
    #name = txt_filename.get("1.0", tk.END + "-1c")
    name = "rt60.csv"
    with open(name, "w+") as file:

        file.write("{}\n".format(get_header()))
        file.write("{}\n".format(get_param_str()))

        for f_idx, rt in enumerate(glob_rts):

            freq = glob_freqs[f_idx]
            file.write("{},{}\n".format(freq, rt))

def write_wav():
    import scipy.io.wavfile as wav
    data = glob_data

    wav.write("out.wav", SAMPLERATE, data)

def get_header() -> str:
    import datetime
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")

def get_param_str() -> str:

    return "{}, {}".format(nfft_val.get(), nperseg_val.get())

class Value:

    def on_inc(self):
        if self.mode == "linear":
            self.val.set(self.val.get() + 1)
        else:
            self.val.set(int(self.val.get() * 2))
        draw(glob_data)
        

    def on_dec(self):
        if self.mode == "linear":
            self.val.set(self.val.get() - 1)
        else:
            self.val.set(int(self.val.get() / 2))
        draw(glob_data)

    def get(self):
        return self.val.get()

        

    def __init__(self, m: Misc, r: int, var, text: str, mode: str = "linear") -> None:
        self.val = var
        #self.val.set(256)
        self.mode = mode

        self.lb_text = tk.Label(master=m, text=text, width=10)
        self.lb_text.grid(row=r, column=0)
    
        self.btn_inc = tk.Button(master=m, text="+", command=self.on_inc, width=10)
        self.btn_inc.grid(row=r, column=2)

        self.btn_dec = tk.Button(master=m, text="-", command=self.on_dec, width=10)
        self.btn_dec.grid(row=r, column=3)

        self.lb_value = tk.Label(master=m, textvariable=self.val, width=10)
        self.lb_value.grid(row=r, column=1)
    
def draw(data):
    calc()(data, nfft_val.get(), nperseg_val.get(), max_mse.get(), max_slope.get(), thresh.get())

def main(data, window, close_call):
    global nfft_val, nperseg_val, max_mse_val, max_mse, max_slope, thresh, axis, glob_data, glob_freqs, glob_rts, txt_filename, all_rts


    #glob_rts = None
    fig, axis = plt.subplots(1, 4)

    #window = tk.Tk()

    nfft = tk.IntVar(master=window)
    nfft.set(2048)
    nfft_val = Value(window, 0, nfft, "nfft", mode="")

    nperseg = tk.IntVar(master=window)
    nperseg.set(64)
    nperseg_val = Value(window, 1, nperseg, "nperseg", mode="")

    max_mse = tk.IntVar(master=window)
    max_mse.set(60)
    max_mse_val = Value(window, 2, max_mse, "maximaler Fehler")


    max_slope = tk.IntVar(master=window)
    max_slope.set(-20)
    max_slope_val = Value(window, 3, max_slope, "maximaler Anstieg")

    thresh = tk.IntVar(master=window)
    thresh.set(0)
    thresh_val = Value(window, 4, thresh, "Schwelle für RT Messung [dB]")


    btn_writefile = tk.Button(master=window, text="write rt to file", command=write_file)
    btn_writefile.grid(row=5, column=0)

    btn_rec = tk.Button(master=window, text="REC", command=close_call)
    btn_rec.grid(row=5, column=3)

    txt_filename = tk.Text(master=window, width=20, height=1)
    txt_filename.grid(row=5, column=1)
    txt_filename.insert("1.0", "out.csv")

    btn_wav = tk.Button(master=window, text="save sample as WAV", command=write_wav)
    btn_wav.grid(row=6, column=1)
    #data = data
    glob_data = data

    draw(data)
    #window.mainloop()

if __name__ == "__main__":

    import audiofile

    data, sr = audiofile.read("data/wav1.wav")
    print(sr)

    #data = data[1110100:1140600]
    data = data[2075000:2087500]

    main(data)
    
    








    
