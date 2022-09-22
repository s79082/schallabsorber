import queue as qu
import sys
from tkinter import S
import numpy as np
import sounddevice as sd

from tmp import to_dB

LISTENING = "listening"
RECORDING = "recording"

class AudioInput:

    q = qu.Queue()

    def __init__(self, channels, samplerate) -> None:
        self.mapping: list = [c - 1 for c in channels]  # Channel numbers start with 1
        self.start = None
        self.end = 0
        self.n_frames = 0
        length = int(200 * samplerate / (1000 * 10))

        self.record_data = np.zeros((length, len(channels)))

        self.state = LISTENING

        self.stream = sd.InputStream(
            device=None, channels=max(channels),
            samplerate=samplerate, callback=self._audio_callback)

        with self.stream as s:
            input()

    def get_data(self) -> np.ndarray:
        pass

    def get_data_length(self) -> int:
        pass

    def _audio_callback(self, indata, frames, _, status):
        if status:
            print(status, file=sys.stderr)

        # decode channels
        data = indata[:, self.mapping]

        # linearize
        data_dB = to_dB(data)

        # get max amp
        max_val = np.max(data_dB)

        if max_val > 85:

            if self.state == LISTENING:

                shift = len(data)

                # TODO start recording
                #self.record_data = np.roll(self.record_data, -shift, axis=0)

                pass

            else:
                pass

        elif self.state ==  LISTENING:
            # still recording
            self.record_data = np.roll(self.record_data, -shift, axis=0)
            self.record_data[-shift:, :] = data

        
        if max_val < 70:

            if self.state == RECORDING:
                # TODO finish recording
                pass

        


        # check for interest
        # TODO own class/file
        # if max_val > 85 and start is None:
        #     print("START")
        #     self.start = i
        #     n_frames = 0
        #     return
        # if start is not None and max_val < 70:
        #     print("END")
        #     #print(start, end)
        #     print(n_frames / 44100.0)
        #     n_frames = 0
        #     start = None
        #     draw = True
        #     #print(len(record_data) / 44100)
        # elif start is not None:
        #     shift = len(data)
        #     #print(shift)
        #     record_data = np.roll(record_data, -shift, axis=0)
        #     #print(record_data[-shift:, :])
        #     #print(data[0])
        #     record_data[-shift:, :] = data

        print(len(data))


if __name__ == "__main__":
    dev = AudioInput([1], 44100)


