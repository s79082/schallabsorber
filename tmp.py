from typing import Callable
from pyparsing import col
from scipy.io.wavfile import read
import audiofile
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal.windows as win

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

def calc():

    def f(data, nfft, nperseg, max_mse, max_slope, thresh):

        global axis, glob_rts, glob_freqs

        for a in axis:
            a.clear()

        #nfft = int(txt_nfft.get("1.0", tk.END))
        #nperseg = int(txt_nperseg.get("1.0", tk.END))
        print(nfft, nperseg)
        #data = data[280000:360000]
        print(type(data))

        #data, sr = audiofile.read("data/wav1.wav")
        #data, sr = audiofile.read("D:/t2-exp.wav")

        #data = data[280000:360000]
        print(data.shape)
        tmp_data = np.zeros((data.shape[0],))
        for idx, _ in enumerate(tmp_data):
            tmp_data[idx] = data[idx]

        data = tmp_data

        print(type(data), len(data))
        #print(data)
        #plt.plot(data)
        #plt.show()
        #interval = detect_intervals(data)
        interval = [(0, len(data))]
        matrices = []

        sr = 44100

        #rt_acc = np.zeros((480,), dtype=np.float64)

        import scipy.signal as sig

        acc_mat = None
        init_acc = True
        n_mat = 0

        # for each interest
        for _, end in interval:

            #end += 30000
            #data_slice = data[end - 16000 : end]
            data_slice = data
            #print(len(data_slice))
            

            #cutoff = len(data_slice) - 50000

            #print(type(nfft), type(nperseg))
            f, ts, M = sig.spectrogram(data_slice, fs=sr, window="hann", nfft=nfft, nperseg=nperseg)
            #plt.pcolormesh(t, f, M)
            #plt.show()

            #index_0_04 = np.where(f == 0.3)[0][0]

            #tmp = M[index_0_04]

            #plt.scatter(t, tmp)
            #plt.show()

            #print(index_0_04)

            print(M.shape)
            print(len(f))
            print(len(ts))

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
        acc_mat /= len(interval)
        # print(f[2])
        # plt.plot(acc_mat[2])
        # plt.show()
        # print(f[7])
        # plt.plot(acc_mat[7])
        # plt.show()
        print("len M", len(M))
        M_tmp = np.zeros((M.shape[0], M.shape[1],), dtype=np.float32)
        for r, row in enumerate(M_tmp):
            for c, col in enumerate(row):
                M_tmp[r, c] = col

        print(type(M_tmp[0, 0]), M_tmp.shape)

        #M_tmp = np.swapaxes(M_tmp, 0, 1)
        print(len(ts), len(f))
        #axis[0].pcolormesh(ts, f, M_tmp)
        axis[0].plot(to_dB(data_slice))
        axis[0].set_xlabel("time (s)")
        axis[0].set_ylabel("Pegel (dB)")
        #plt.pcolormesh(ts, f, acc_mat)
        #plt.show()

        M_db = to_dB(acc_mat)

        #plt.pcolormesh(t, f, M_db, shading="gouraud")
        
        #plt.pcolormesh(ts, f, M_db)
        axis[1].pcolormesh(ts, f, M_db)
        axis[1].set_xlabel("time (s)")
        axis[1].set_ylabel("frequency (Hz)")
        #axis[1].colorbar()
        #plt.colorbar(cax=axis[1])
        #plt.ylabel("Frequency (Hz)")
        #plt.xlabel("Time (s)")
        #plt.colorbar(label="SPL (dB)")

        #plt.show()

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

            #if 850 < f[f_idx] < 1150:
            #    plt.plot(samples_over_time)
            # convert to dB
            dBs = to_dB(samples_over_time)
            if 850 < f[f_idx] < 1150 or 4000 < f[f_idx] < 4500:
                #plt.plot(dBs)
                #plt.show()
                #plt.plot(dBs)
                #plt.show()
                print("max", max(dBs))

            if max(dBs) < thresh:
                rts[f_idx] = 0 
                continue
            #if 850 < f[f_idx] < 1150:
            #    plt.plot(dBs)
            #    print(f[f_idx])
                #plt.show()
            #if f_idx == 1:
                #plt.plot(dBs)
                #plt.show()

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
            #if 850 < f[f_idx] < 1150:
            #    print(t_predict, mse, slope)
            #    plt.plot(predicted_dBs)
            #    plt.show()
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
        print(rts[-1:])
        axis[2].scatter(f, rts, s=5)
        axis[2].set_xlabel("frequency (Hz)")
        axis[2].set_ylabel("RT20 (s)")


        #axis[2].plot(f, slopes)
        plt.show()

        print(len(f), f[1] - f[0])

        glob_rts = rts
        glob_freqs = f

        
    
    return f


def write_file(name: str, rts: np.ndarray, freqs: np.ndarray):
    def f():
        with open(name, "w+") as file:

                for f_idx, rt in enumerate(rts):

                    freq = freqs[f_idx]

                    file.write("{},{}\n".format(freq, rt))

    return f

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

def main(data):
    global nfft_val, nperseg_val, max_mse_val, max_mse, max_slope, thresh, axis, glob_data, glob_freqs, glob_rts


    glob_rts = None
    fig, axis = plt.subplots(1, 3)

    window = tk.Tk()

    nfft = tk.IntVar(master=window)
    nfft.set(256)
    nfft_val = Value(window, 0, nfft, "nfft", mode="")

    nperseg = tk.IntVar(master=window)
    nperseg.set(256)
    nperseg_val = Value(window, 1, nperseg, "nperseg", mode="")

    max_mse = tk.IntVar(master=window)
    max_mse.set(60)
    max_mse_val = Value(window, 2, max_mse, "maximaler Fehler")


    max_slope = tk.IntVar(master=window)
    max_slope.set(-20)
    max_slope_val = Value(window, 3, max_slope, "maximaler Anstieg")

    thresh = tk.IntVar(master=window)
    thresh.set(50)
    thresh_val = Value(window, 4, thresh, "Schwelle für RT Messung [dB]")


    #btn_writefile = tk.Button(master=window, text="write rt to file", command=write_file("test.out", glob_rts, glob_freqs))
    #btn_writefile.grid(row=5, column=0)

    txt_filename = tk.Text(master=window)
    txt_filename.grid(row=5, column=1)
    txt_filename.insert("1.0", "out.csv")
    #data = data
    glob_data = data
    draw(data)
    window.mainloop()

if __name__ == "__main__":
    main(None)
    
    








    
