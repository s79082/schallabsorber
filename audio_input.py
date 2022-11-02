import queue as qu
from re import M
import sys
from tkinter import DISABLED, S
import tkinter
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
        print(type(max_val))

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

        print(length)

        self.window = tkinter.Tk()

        self.btn_rec_var = tkinter.StringVar(master=self.window, value="DISABLED")

        self.btn_rec = tkinter.Button(master=self.window, textvariable=self.btn_rec_var, command=self._toggle_rec)
        self.btn_rec.grid(row=0, column=0)

        self.btn_wav = tkinter.Button(master=self.window, text="load WAV file", command=self.load_file)
        self.btn_wav.grid(row=1, column=0)



        self.record_data = np.zeros((length * 100, len(channels)))

        self.state = States.DISABLED


        self.stream = sd.InputStream(
            device=None, channels=max(channels),
            samplerate=samplerate, callback=self._audio_callback)

        with self.stream as s:
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
        data, _ = audiofile.read("data/2022-09-24_SLM_001_Audio_FS129.7dB(PK)_00.wav")
        #data2, _ = audiofile.read("data/2022-09-24_SLM_002_Audio_FS129.7dB(PK)_00.wav")
        #data = np.concatenate((data, data2))

        intervals = scan(data)

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
            #print("soos")
            data = indata

        # linearize
        data_dB = to_dB(data)

        # get max amp
        max_val = np.max(data_dB)

        shift = len(data)

        if max_val > 85:

            if self.state == States.LISTENING:

                # TODO start recording
                print("start")
                #self.record_data = np.roll(self.record_data, -shift, axis=0)
                #self.record_data[-shift:, :] = dxata
                #self.record_data = np.roll(self.record_data, -shift, axis=0)
                self.state = States.RECORDING
                self.n_frames = 0
                pass

            else:
                pass

        if self.state ==  States.RECORDING:
            # still recording
            #print("still rec")
            self.record_data = np.roll(self.record_data, -shift, axis=0)
            self.record_data[-shift:, :] = data
            self.n_frames += frames
        
        if max_val < 70:

            if self.state == States.RECORDING:
                # TODO finish recording
                self.state = States.LISTENING
                print("stop")
                print(self.n_frames / self.samplerate)
                
                data_slice = np.array([ x for x in self.record_data if x != 0 ])

                self.state = States.DISABLED
                
                main(data_slice[len(data_slice) - 15000:], self.window, self._toggle_rec)

    def plot(self, data: np.ndarray):

        fig, axis = plt.subplots(1, 3, figsize=(10, 10))

        time_axis = np.arange(len(data)) / SAMPLERATE

        axis[0].plot(time_axis, to_dB(data))
        axis[0].set_xlabel("time (s)")
        axis[0].set_ylabel("Pegel (dB)")

        spec = get_spectogram(data, {"nfft": 2048, "nperseg": 64, "sr": 48000})

        val_dbs = to_dB(spec.values)

        axis[1].pcolormesh(spec.times, spec.frequencies, val_dbs)
        axis[1].set_xlabel("time (s)")
        axis[1].set_ylabel("frequency (Hz)")

        fs, rts, _, _ = calc_rt(spec, {"thresh": 0})

        axis[2].scatter(fs, rts, s=5)
        axis[2].set_xlabel("frequency (Hz)")
        axis[2].set_ylabel("RT60 (s)")
        
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
    
        mses[f_idx] = mean_squared_error(amps_db, amps_predicted)

        slopes[f_idx] = slope

        if t_predict > 0:
            rts[f_idx] = t_predict

        
        return (fs, rts, mses, slopes)

if __name__ == "__main__":
    dev = AudioInput([1], 48000)
    print(dev.get_data())







