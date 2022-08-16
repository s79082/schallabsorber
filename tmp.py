from scipy.io.wavfile import read
import audiofile
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal.windows as win

from fft_util import calc_fft, generate_test_sine
from reverberation_detector import detect_intervals, interval_len

from sklearn.metrics import mean_squared_error

import tkinter as tk
from tkinter import END, ttk
    
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

# read the data
#data, sr = audiofile.read("data/wav1.wav")

# cut away a part of interest
#data = data[INTERVAL[0] : INTERVAL[1]]

def main():

    def f():

        global txt_nfft, txt_nperseg, axis

        for a in axis:
            a.clear()

        nfft = int(txt_nfft.get("1.0", tk.END))
        nperseg = int(txt_nperseg.get("1.0", tk.END))
        print(nfft, nperseg)

        data, sr = audiofile.read("data/wav1.wav")
        interval = detect_intervals(data)
        #interval = [(0, len(data))]
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
        # contains all rts, not dep. on mses
        rts_no_mse = np.zeros(f.shape)
        slopes = np.zeros(f.shape)

        #rts = []

        # samples for each freq
        for f_idx, samples_over_time in enumerate(acc_mat):

            # convert to dB
            dBs = to_dB(samples_over_time)

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

            if mse <= 20 and t_predict > 0:
                rts[f_idx] = t_predict  




    #plt.plot(f, rts_no_mse)log
        #axis[2].plot(f, rts)
        #axis[2].plot(f, mses)

        #plt.plot(f, slopes)
        #plt.legend(["rt", "mse", "rt after mse", "slope"])

        f = f[:175]
        print(f)
        rts = rts[:175]
        axis[2].plot(f, rts)
        plt.show()

    
    return f


if __name__ == "__main__":
    # load file
    
    #data = generate_test_sine(SAMPLERATE)

    fig, axis = plt.subplots(1, 3)

    window = tk.Tk()

    txt_nfft = tk.Text(master=window, height=1, width=10)
    txt_nfft.insert("1.0", "248")
    txt_nfft.pack()

    def change_listener_nfft():
        val = int(float(txt_nfft.get("1.0")))
        print(val)
        val *= 2

        txt_nfft.delete("1.0", END)
        txt_nfft.insert("1.0", val)

        main()()

    btn_inc_nfft = tk.Button(master=window, text="+", command=change_listener_nfft)
    btn_inc_nfft.pack()
    
    def change_listener_sld_nfft(val):
        val = int(float(val))
        txt_nfft.delete("1.0", END)
        txt_nfft.insert("1.0", val)
        
        main()()

    sld_nfft = ttk.Scale(master=window, from_=24, to=SAMPLERATE, command=change_listener_sld_nfft)
    sld_nfft.pack()

    txt_nperseg = tk.Text(master=window, height=1, width=10)
    txt_nperseg.insert("1.0", "248")    
    txt_nperseg.pack()

    btn_draw = tk.Button(master=window, text="calc", command=main())
    btn_draw.pack()

    


    

    # detect interests
    

#plt.plot()

    #rt_acc /= 4

    #plt.plot(rt_acc)
    #plt.show()


    window.mainloop()








    
