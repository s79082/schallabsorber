import queue as qu
from re import M
import sys
from tkinter import DISABLED, S
import tkinter as tk
import numpy as np
from sklearn.metrics import mean_squared_error
import sounddevice as sd
import scipy.signal as sig


import matplotlib.pyplot as plt

from tmp import DB_DIFF, SAMPLERATE, to_dB, main
class States:
    LISTENING = "listening"
    RECORDING = "recording"
    DISABLED = "disabled"


def scan(data: np.ndarray) -> list[tuple[int, int]]:

    frame_size = 1136
    min_interval_len = SAMPLERATE

    split_data = np.array_split(data, int(len(data) / frame_size))
    state = "listen"

    intervals = []
    start = stop = None

    for idx, frame in enumerate(split_data):
        data_dB = to_dB(frame)

        # get max amp
        max_val = np.max(data_dB)

        if max_val > 69 and state == "listen":
            state = "record"
            start = idx * frame_size

        if max_val < 55 and state == "record":

            stop = idx * frame_size
            if stop - start > min_interval_len:
                intervals.append((start, stop))
            start = stop = None
            state = "listen"

    return intervals

class AudioInput:

    q = qu.Queue()

    def __init__(self, channels, samplerate) -> None:
        self.mapping: list = [c - 1 for c in channels]  # Channel numbers start with 1
        self.start = None
        self.end = 0
        self.n_frames = 0
        length = int(200 * samplerate / (1000 * 10))

        self.samplerate = samplerate

        self.acc_rts = None
        self.n_rts = 0

        print(length)

        self.window = tk.Tk()

        self.btn_rec_var = tk.StringVar(master=self.window, value="DISABLED")
        self.btn_rec = tk.Button(master=self.window, textvariable=self.btn_rec_var, command=self._toggle_rec)
        self.btn_rec.grid(row=6, column=0)

        self.btn_wav = tk.Button(master=self.window, text="load WAV file", command=self.load_file)
        self.btn_wav.grid(row=7, column=1)

        self.nfft = tk.IntVar(master=self.window)
        self.nfft.set(2048)
        self.nfft_val = Value(self.window, 0, self.nfft, "nfft", mode="")

        self.nperseg = tk.IntVar(master=self.window)
        self.nperseg.set(64)
        self.nperseg_val = Value(self.window, 1, self.nperseg, "nperseg", mode="")

        max_mse = tk.IntVar(master=self.window)
        max_mse.set(60)
        max_mse_val = Value(self.window, 2, max_mse, "maximaler Fehler")

        max_slope = tk.IntVar(master=self.window)
        max_slope.set(-20)
        max_slope_val = Value(self.window, 3, max_slope, "maximaler Anstieg")

        thresh = tk.IntVar(master=self.window)
        thresh.set(-10)
        thresh_val = Value(self.window, 4, thresh, "Schwelle für RT Messung [dB]")


        btn_writefile = tk.Button(master=self.window, text="write rt to file", command=None)
        btn_writefile.grid(row=5, column=0)

        btn_rec = tk.Button(master=self.window, text="REC", command=None)
        btn_rec.grid(row=5, column=3)

        txt_filename = tk.Text(master=self.window, width=20, height=1)
        txt_filename.grid(row=5, column=1)
        txt_filename.insert("1.0", "out.csv")

        btn_wav = tk.Button(master=self.window, text="save sample as WAV", command=None)
        btn_wav.grid(row=6, column=1)

        self.record_data = np.zeros((length * 100, len(channels)))

        self.state = States.DISABLED

        try:
            self.stream = sd.InputStream(
                device=None, channels=max(channels),
                samplerate=samplerate, callback=self._audio_callback)

            with self.stream as s:
                self.window.mainloop()
        except:
            self.stream = None
            self.window.mainloop()

        


    def _toggle_rec(self):

        if self.state == States.DISABLED:
            plt.close()
            self.record_data = np.zeros(self.record_data.shape)
            self.state = States.LISTENING
            self.btn_rec_var.set(self.state)

        elif self.state == States.LISTENING or self.state == States.RECORDING:

            self.state = States.DISABLED
            self.btn_rec_var.set(self.state)

    def load_file(self):
        import audiofile
        data, _ = audiofile.read("D:/Downloads/TransferXL-08j50XPJzvhhC9/Messungen 21_11_2022/TestN333-1/2022-11-23_SLM_000_Audio_FS129.7dB(PK)_00.wav")
        #data, _ = audiofile.read("data/2022-09-24_SLM_001_Audio_FS129.7dB(PK)_00.wav")
        #data2, _ = audiofile.read("data/2022-09-24_SLM_002_Audio_FS129.7dB(PK)_00.wav")
        #data = np.concatenate((data, data2))

        intervals = scan(data)

        self.n_intervals = len(intervals)

        for start, stop in intervals:
            self.plot(data[stop - 15000 : stop])


    def get_data(self) -> np.ndarray:
        return self.record_data

    def _audio_callback(self, indata, frames, _, status):
        if status:
            print(status, file=sys.stderr)

        # decode channels
        if len(indata.shape) > 1: 
            data = indata[:, self.mapping]

        else:
            data = indata

        # linearize
        data_dB = to_dB(data)

        # get max amp
        max_val = np.max(data_dB)

        shift = len(data)

        if max_val > 85:

            if self.state == States.LISTENING:

                # start recording
                print("start")
                self.state = States.RECORDING
                self.n_frames = 0
                pass

            else:
                pass

        if self.state ==  States.RECORDING:
            # still recording
            
            self.record_data = np.roll(self.record_data, -shift, axis=0)
            self.record_data[-shift:, :] = data
            self.n_frames += frames
        
        if max_val < 70:

            if self.state == States.RECORDING:
                # finish recording
                self.state = States.LISTENING
                print("stop")
                print(self.n_frames / self.samplerate)
                
                # remove zero values
                data_slice = np.array([ x for x in self.record_data if x != 0 ])

                self.state = States.DISABLED
                
                self.plot(data_slice)

    def plot(self, data: np.ndarray):

        fig, axis = plt.subplots(1, 4, figsize=(10, 10))
        fig.canvas.manager.full_screen_toggle()

        time_axis = np.arange(len(data)) / SAMPLERATE

        axis[0].plot(time_axis, to_dB(data))
        axis[0].set_xlabel("time (s)")
        axis[0].set_ylabel("Pegel (dB)")

        spec = get_spectogram(data, {"nfft": self.nfft.get(), "nperseg": self.nperseg.get(), "sr": 48000})

        val_dbs = to_dB(spec.values)

        axis[1].pcolormesh(spec.times, spec.frequencies, val_dbs)
        axis[1].set_xlabel("time (s)")
        axis[1].set_ylabel("frequency (Hz)")

        fs, rts, _, _ = calc_rt(spec, {"thresh": -20})

        print(rts)

        fs = fs[:300]
        rts = rts[:300]

        if self.acc_rts is None:
            self.acc_rts = np.zeros(fs.shape)
        self.acc_rts += rts
        self.n_rts += 1

        #plt.scatter(fs, self.acc_rts / self.n_rts, s=5)
        #plt.show()
        print("length", len(fs))

       

        axis[2].scatter(fs, rts, s=5)
        axis[2].set_xlabel("frequency (Hz)")
        axis[2].set_ylabel("RT60 (s)")
        #axis[2].scatter([250, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000],
        #    [0.98, 0.53, 0.82, 0.8, 0.6, 0.71, 0.69, 0.63, 0.68, 0.64, 0.63, 0.63, 0.63], c="green", s=5)

        axis[3].scatter(fs, self.acc_rts / self.n_rts, s=5)
        axis[3].set_xlabel("frequency (Hz)")
        axis[3].set_ylabel("avg RT60 (s) [of {} samples{}] ".format(str(self.n_rts), " FINAL" if self.n_rts == self.n_intervals else ""))
        
        fig.show()
class Spectrogram:
    frequencies: np.ndarray
    times: np.ndarray
    values: np.ndarray

def get_spectogram(data: np.ndarray, args: dict) -> Spectrogram:
    
    spec = Spectrogram()

    spec.frequencies, spec.times, spec.values = sig.spectrogram(data, fs=args["sr"], window="hann", nfft=args["nfft"], nperseg=args["nperseg"])

    return spec

def calc_rt(spec: Spectrogram, args: dict)       ->     tuple:

    fs, ts, M = spec.frequencies, spec.times, spec.values

    rts = np.zeros(fs.shape)
    mses = np.zeros(fs.shape)
    slopes = np.zeros(fs.shape)
    
    for f_idx, amps in enumerate(M):

        amps_db = to_dB(amps)

        if np.max(amps_db) < args["thresh"]:
            continue

        # linear regression 
        model = np.polyfit(ts, amps_db, 1)

        # get linear parameters     
        slope = model[0]
        intersect = model[1]

        # amp we want 
        y_end = intersect - DB_DIFF

        # get predicted timestep
        t_predict = (y_end - intersect) / slope

        # calc mse for linear check
        amps_predicted = slope * ts + intersect
    
        # set mse
        mses[f_idx] = mean_squared_error(amps_db, amps_predicted)

        # set slopes
        slopes[f_idx] = slope

        # set rt
        if t_predict > 0:
            rts[f_idx] = t_predict

    return (fs, rts, mses, slopes)

class Value:

    def on_inc(self):
        if self.mode == "linear":
            self.val.set(self.val.get() + 1)
        else:
            self.val.set(int(self.val.get() * 2))
        
        #draw(glob_data)
        

    def on_dec(self):
        if self.mode == "linear":
            self.val.set(self.val.get() - 1)
        else:
            self.val.set(int(self.val.get() / 2))
        #draw(glob_data)

    def get(self):
        return self.val.get()

        

    def __init__(self, m: tk.Misc, r: int, var, text: str, mode: str = "linear") -> None:
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

if __name__ == "__main__":
    dev = AudioInput([1], 48000)
    print(dev.get_data())







