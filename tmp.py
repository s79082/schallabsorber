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


DB_DIFF = 20.0 

def to_dB(data: np.ndarray) -> np.ndarray:
    return 10 * np.log10(abs(data) / X_0)

SAMPLERATE = 48000

INTERVAL = (1115000, 1200000)
INTERVAL_LEN = INTERVAL[1] - INTERVAL[0]

# amount of frames we split the data
N_FRAMES = 50

def main():

    def f(nfft, nperseg, max_mse, max_slope, thresh):

        global txt_nfft, txt_nperseg, axis

        for a in axis:
            a.clear()

        #nfft = int(txt_nfft.get("1.0", tk.END))
        #nperseg = int(txt_nperseg.get("1.0", tk.END))
        print(nfft, nperseg)

        #data, sr = audiofile.read("data/wav1.wav")
        data, sr = audiofile.read("E:/t2-exp.wav")
        data = data[280000:360000]
        #plt.plot(data)
        #plt.show()
        #interval = detect_intervals(data)
        interval = [(0, len(data))]
        matrices = []

        SAMPLERATE = sr

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
            print(len(data_slice))
            

            #cutoff = len(data_slice) - 50000

            print(type(nfft), type(nperseg))
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

        axis[0].pcolormesh(ts, f, acc_mat)
        #plt.pcolormesh(ts, f, acc_mat)
        #plt.show()

        M_db = 10 * np.log10(abs(acc_mat) / X_0)

        #plt.pcolormesh(t, f, M_db, shading="gouraud")
        
        #plt.pcolormesh(ts, f, M_db)
        axis[1].pcolormesh(ts, f, M_db)
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
        axis[2].scatter(f, rts, s=5)
        #axis[2].plot(f, slopes)
        plt.show()

    
    return f
def change_nfft(operation: Callable):
    def f():
        global NFFT
        val = NFFT
        val = operation(val)
        #val *= 2

        txt_nfft.delete("1.0", END)
        txt_nfft.insert("1.0", val)

        NFFT = val

        main()()

    return f


def increment_nfft():
    #val = int(float(txt_nfft.get("1.0")))
    global NFFT
    val = NFFT
    val *= 2

    txt_nfft.delete("1.0", END)
    txt_nfft.insert("1.0", val)

    NFFT = val

    main()()

def decrement_nfft():
    global NFFT
    val = NFFT
    val = int(val / 2)

    txt_nfft.delete("1.0", END)
    txt_nfft.insert("1.0", val)

    NFFT = val

    main()()



class Value:

    def on_inc(self):
        if self.mode == "linear":
            self.val.set(self.val.get() + 1)
        else:
            self.val.set(int(self.val.get() * 2))
        draw()
        

    def on_dec(self):
        if self.mode == "linear":
            self.val.set(self.val.get() - 1)
        else:
            self.val.set(int(self.val.get() / 2))
        draw()

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
    
def draw():
    main()(nfft_val.get(), nperseg_val.get(), max_mse.get(), max_slope.get(), thresh.get())

if __name__ == "__main__":
    # load file
    
    NFFT = 256

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
    max_mse_val = Value(window, 2, max_mse, "max_mse")


    max_slope = tk.IntVar(master=window)
    max_slope.set(-20)
    max_slope_val = Value(window, 3, max_slope, "slope")

    thresh = tk.IntVar(master=window)
    thresh.set(50)
    thresh_val = Value(window, 4, thresh, "thresh [dB]")

    # txt_nfft = tk.Text(master=window, height=1, width=10)
    # txt_nfft.insert("1.0", "248")
    # txt_nfft.grid(row=0, column=0)

    # btn_inc_nfft = tk.Button(master=window, text="+", command=increment_nfft)
    # btn_dec_nfft = tk.Button(master=window, text="-", command=decrement_nfft)
    # btn_inc_nfft.grid(row=0, column=1)
    # btn_dec_nfft.grid(row=0, column=2)
    
    # def change_listener_sld_nfft(val):
    #     val = int(float(val))
    #     txt_nfft.delete("1.0", END)
    #     txt_nfft.insert("1.0", val)
        
    #     main()()

    # sld_nfft = ttk.Scale(master=window, from_=24, to=SAMPLERATE, command=change_listener_sld_nfft)
    # #sld_nfft.pack()

    # txt_nperseg = tk.Text(master=window, height=1, width=10)
    # txt_nperseg.insert("1.0", "248")    
    # #txt_nperseg.pack()

    # btn_draw = tk.Button(master=window, text="calc", command=main())
    # btn_draw.grid(row=3, column=0)

    draw()
    window.mainloop()








    
