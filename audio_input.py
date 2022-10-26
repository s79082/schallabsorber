import queue as qu
import sys
from tkinter import DISABLED, S
import tkinter
import numpy as np
import sounddevice as sd

import matplotlib.pyplot as plt

from tmp import to_dB, main
class States:
    LISTENING = "listening"
    RECORDING = "recording"
    DISABLED = "disabled"

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
        data = data[int(1.29 * 10 ** 6): int(1.30 * 10 **6)]
        main(data, self.window, self._toggle_rec)
        l = len(data)
        data, _ = audiofile.read("data/2022-09-24_SLM_002_Audio_FS129.7dB(PK)_00.wav")
        data = data[int(1.268 * 10 ** 6): int(1.268 * 10 ** 6) + l]

        main(data, self.window, self._toggle_rec)




    def get_data(self) -> np.ndarray:
        return self.record_data

    def get_data_length(self) -> int:
        return len(self.record_data)

    def _audio_callback(self, indata, frames, _, status):
        if status:
            print(status, file=sys.stderr)

        # decode channels
        data = indata[:, self.mapping]

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
                #self.record_data[-shift:, :] = data
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
                
                main(data_slice, self.window, self._toggle_rec)

        #print(len(data))


if __name__ == "__main__":
    dev = AudioInput([1], 44100)
    print(dev.get_data())


