import queue as qu
from re import M
import sys
from tkinter import DISABLED, S
import tkinter as tk
import numpy as np
from sklearn.metrics import mean_squared_error
import sounddevice as sd
import scipy.signal as sig

import color_manager as color
from interval_row import IntervalRow

from matplotlib.backend_bases import MouseButton

import matplotlib.pyplot as plt

from tmp import DB_DIFF, SAMPLERATE, to_dB, main
class States:
    LISTENING = "listening"
    RECORDING = "recording"
    DISABLED = "disabled"

class Meassurement:

    rts : np.ndarray
    fs : np.ndarray
    mses: np.ndarray
    slope: np.ndarray

    graph_color: str

    def __init__(self, start, stop) -> None:
        self.start = start
        self.stop = stop

        self.graph_color = color.get_color()

    def __str__(self) -> str:
        return "({}, {})".format(self.start, self.stop)

    def is_calculated(self):
        return self.rts is not None and self.fs is not None

    def calculate(self, data: np.ndarray, audio_input_obj):
        self.fs, self.rts, self.mses, _ = calc_rt(get_spectogram(data[self.start : self.stop], {"nfft": audio_input_obj.nfft.get(), "nperseg": audio_input_obj.nperseg.get(), "sr": 48000}), {"thresh": -20})



def scan(data: np.ndarray) -> list[tuple[int, int]]:

    frame_size = int(1136 / 2)
    min_interval_len = SAMPLERATE

    split_data = np.array_split(data, int(len(data) / frame_size))
    state = "listen"

    intervals = []
    # start = stop = None

    maximas = []

    # _data = []

    for idx, frame in enumerate(split_data):
        data_dB = to_dB(frame)

        # get max amp
        max_val = np.max(data_dB)
        maximas.append(max_val)

        if max_val > 69 and state == "listen":
            state = "record"
            start = idx * frame_size

        if max_val < 55 and state == "record":

            stop = idx * frame_size
            if stop - start > min_interval_len:
                #intervals.append((start, stop))
                intervals.append((stop - SAMPLERATE, stop))
                pass
            start = stop = None
            state = "listen"


    return intervals

    #tmp = np.diff(maximas)
    #plt.plot(np.diff(maximas))
    #plt.show()
    #plt.plot(np.diff(np.diff(maximas)))
    #plt.show()
    #plt.plot(np.diff(np.diff(np.diff(maximas))))
    #plt.show()

    # state = "over"
    # print(len(tmp))

    # for idx, diff in enumerate(tmp):

    #     print("lol", idx)

    #     if diff < 0 and state == "over":
    #         print(diff)
    #         state = "under"
    #         start = idx * frame_size

    #     if max_val > 0 and state == "under":

    #         print("over")
    #         stop = idx * frame_size
    #         if stop - start > min_interval_len:
    #             intervals.append((start, stop))
    #         start = stop = None
    #         state = "over"
    #print(intervals)

    #interval_len = int(0.016910 * (10^6))
    interval_len = 20000
    start = 2403090
    #start = int(2.403090 * (10^6))
    stop = start + interval_len

    #for i in range(10):

    _20_sec = SAMPLERATE * 20

    #tmp = [(start, stop), (start + 960000, stop + 960000)]

    tmp = []
    for i in range(10):
        print(i)
        tmp.append((start + i * _20_sec, stop + i *  _20_sec))

    print(tmp)

    return tmp

class AudioInput:

    def __init__(self, channels, samplerate) -> None:
        self.mapping: list = [c - 1 for c in channels]  # Channel numbers start with 1
        self.start = None
        self.stopp = None
        self.end = 0
        self.n_frames = 0
        length = int(200 * samplerate / (1000 * 10))

        self.samplerate = samplerate

        self.acc_rts = None
        self.n_rts = 0

        self.all_rts = []

        self.measurenents = []

        print(length)

        self.window = tk.Tk()

        def ondraw():
            
            plt.cla()
            for mes, var in self.show.items():
            #for mes in self.measurenents:
                if var.get():
                    mes.calculate(self.data, self)
                    plt.plot(mes.fs, mes.rts)

            plt.show()

        self.btn_rec_var = tk.StringVar(master=self.window, value="DISABLED")
        self.btn_rec = tk.Button(master=self.window, textvariable=self.btn_rec_var, command=self._toggle_rec)
        self.btn_rec.grid(row=6, column=0)

        self.btn_wav = tk.Button(master=self.window, text="load WAV file", command=self.load_file)
        self.btn_wav.grid(row=7, column=1)

        self.nfft = tk.IntVar(master=self.window)
        self.nfft.set(2048)
        self.nfft_val = Value(self.window, 0, self.nfft, "nfft", ondraw, mode="")

        self.nperseg = tk.IntVar(master=self.window)
        self.nperseg.set(64)
        self.nperseg_val = Value(self.window, 1, self.nperseg, "nperseg", ondraw, mode="")

        max_mse = tk.IntVar(master=self.window)
        max_mse.set(60)
        max_mse_val = Value(self.window, 2, max_mse, "maximaler Fehler", ondraw)

        max_slope = tk.IntVar(master=self.window)
        max_slope.set(-20)
        max_slope_val = Value(self.window, 3, max_slope, "maximaler Anstieg", ondraw)

        thresh = tk.IntVar(master=self.window)
        thresh.set(-10)
        thresh_val = Value(self.window, 4, thresh, "Schwelle für RT Messung [dB]", ondraw)


        btn_writefile = tk.Button(master=self.window, text="write rt to file", command=self.on_write_file)
        btn_writefile.grid(row=5, column=0)

        #btn_rec = tk.Button(master=self.window, text="REC", command=None)
        #btn_rec.grid(row=5, column=3)

        #txt_filename = tk.Text(master=self.window, width=20, height=1)
        #txt_filename.grid(row=5, column=1)
        #txt_filename.insert("1.0", "out.csv")

        #btn_wav = tk.Button(master=self.window, text="save sample as WAV", command=None)
        #btn_wav.grid(row=6, column=1)



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

    def draw_rts(self):
        pass

    def on_write_file(self):
        from tkinter.filedialog import asksaveasfilename
        file_name = asksaveasfilename()
        self.write_file(file_name)
        

    def write_file(self, filename: str):
        from datetime import datetime
        fs = self.measurenents[0].fs
        # get rts from messurements

        #selected_messurements = map(lambda m, v: v.get() == True, self.show.items())
        rts = map(lambda m: m.rts, self.measurenents)
        rts_zip = list(zip(*rts))
        with open(filename, mode="w+") as file:
            # header
            file.write("Reveberation time ")
            file.write(str(datetime.now()))
            
            file.write("\n")
            file.write("frequencies (Hz), ")
            rt_header = []
            for i, _ in enumerate(self.measurenents):
                rt_header += ["RT60 #{} (s)".format(str(i))]

            file.write(",".join(rt_header))
            file.write("\n")

            for f, _rts in zip(fs, rts_zip):
                file.write(str(f))
                file.write(",")
                _rts = map(str, _rts)
                file.write(",".join(_rts))
                file.write("\n")

        print(rts_zip[0])
        pass

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
        global start_line, stop_line
        
        fig = plt.gcf()

        def plot_data():
            plt.plot(data_db)
            plt.ylabel("SPL (dB)")


        def redraw_vlines():
            # get start lines
            starts = list(map(lambda m: m.start, self.measurenents))
            stopps = list(map(lambda m: m.stop, self.measurenents))
            start_colors = ["green" for _ in starts]
            stopp_colors = ["red" for _ in stopps]


            positions = starts + stopps
            colors = start_colors + stopp_colors

            ax = plt.gca()

            plt.cla()
            plt.plot(data_db)

            ax.vlines(positions, 0, 90, colors=colors)

            plt.show(block=False)

        win_intervall_select = tk.Tk()
        win_intervall_select.geometry("200x500")

        cid = None
        self.aleady_clicked = False
        def on_add_intervall():
            global cid

            self.btn_add_intervall.config(text="click to abort")
            print(self.aleady_clicked)

            # check if user wants to abort interval selection
            if self.aleady_clicked:
                # ready for new interval input
                fig.canvas.mpl_disconnect(cid)
                self.aleady_clicked = False
                self.btn_add_intervall.config(text="add interval")
                return
            # user clicked btn
            self.aleady_clicked = True
            self.start = None
            self.stopp = None
            def onclick(event):

                ax = plt.gca()
                for l in ax.lines:
                    print(l)
                # for line in ax.lines:
                #    ax.lines.remove(line)
                #x.clear()
                #plt.plot(data_db)
                #plt.show(block=False)

                if event.button == MouseButton.LEFT:
                    self.start = int(event.xdata)
                    # TODO we want the one line (start or stopp) also drawn
                    #ax.vlines([self.start], 0, 90, colors=["orange"])

                elif event.button == MouseButton.RIGHT:
                    self.stopp = int(event.xdata)
                    #ax.vlines([self.stopp], 0, 90, colors=["orange"])
                
                if self.start is not None and self.stopp is not None:

                    # add new messure to be calculated
                    messure = Meassurement(self.start, self.stopp)
                    self.measurenents.append(messure)

                    # add ui element
                    row = IntervalRow(win_intervall_select, messure, self, redraw_vlines)

                    # redraw all vlines from messures
                    redraw_vlines()

                    # ready for next interval input
                    fig.canvas.mpl_disconnect(cid)
                    self.aleady_clicked = False
                    self.btn_add_intervall.config(text="add interval")


                plt.show(block=False)
                print(event.button)

            cid = fig.canvas.mpl_connect('button_press_event', onclick)


        self.btn_add_intervall = tk.Button(master=win_intervall_select, command=on_add_intervall, text="add interval")
        self.btn_add_intervall.pack()

        def on_calculate_rts():

            plt.close()

            for messure in self.measurenents:
                messure.calculate(data, self)

            for gr in self.measurenents:
                plt.plot(gr.fs, gr.rts, color=gr.graph_color)

            plt.ylabel("RT60 (s)")
            plt.xlabel("Frequency (Hz)")
            plt.show(block=False)

            win_intervall_select.destroy()

            win_graphs = tk.Tk()
            win_graphs.geometry("200x500")


            self.show = dict()
            for id, messure in enumerate(self.measurenents):
                self.show[messure] = tk.BooleanVar(master=win_graphs, value=True)
                print(messure.start)
                label = tk.Label(master=win_graphs, text="{}, {}".format(messure.start, messure.stop))
                label.pack()
                color_label = tk.Label(master=win_graphs, text="COLOR", fg=messure.graph_color, bg=messure.graph_color)
                color_label.pack()

                def on_cb_change():

                    plt.cla()

                    for mes, var in self.show.items():
                        if var.get():
                            if not mes.is_calculated():
                                mes.calculate(data, self)

                            plt.plot(mes.fs, mes.rts, color=mes.graph_color)

                    plt.ylabel("RT60 (s)")
                    plt.xlabel("frquency (Hz)")
                    plt.show(block=False)
                cb = tk.Checkbutton(master=win_graphs, variable=self.show[messure], onvalue=True, offvalue=False, command=on_cb_change)
                cb.pack()

        btn_calculate = tk.Button(master=win_intervall_select, text="calculate RTs", command=on_calculate_rts)
        btn_calculate.pack()
            

        from audiofile import read
        from tkinter.filedialog import askopenfilename

        file_name = askopenfilename()


        data, sr = read(file_name)

        self.data = data

        from tkinter.messagebox import askyesno, showinfo

        intervals = scan(data)

        data_db = to_dB(data)

        for _start, _stop in intervals:
            mes = Meassurement(_start, _stop)
            self.measurenents.append(mes)

            IntervalRow(win_intervall_select, mes, self, redraw_vlines)

        redraw_vlines()


        plt.plot(data_db)
        plt.show(block=False)

        return
        xs = list(map(lambda i: i[0], intervals))
        xs += list(map(lambda i: i[1], intervals))
        plt.vlines(xs, ymin=[np.min(data_db)], ymax=[np.max(data_db)], colors=["red"])
        plt.show(block=False)

        def on_select_interval(event):
            for start, stop in intervals:

                if start < event.xdata < stop:

                    print("seleted ", start, stop)
        
        plt.gcf().canvas.mpl_connect('button_press_event', on_select_interval)

        #plt.show()
        self.n_intervals = len(intervals)

        showinfo("", "{} intervals detected".format(len(intervals)))

        for interval_idx, (start, stop) in enumerate(intervals):    

            plt.clf()
            
            # border from the lines 
            frame_border = 10000

            frame_start = start - frame_border
            frame_stop = stop + frame_border

            data_frame = data[frame_start : frame_stop]

            frame_min = np.min(data_frame)
            frame_max = np.max(data_frame)

            frame_len = len(data_frame)

            # the time over the whole file
            global_time = np.arange(start=frame_start, stop=frame_start + frame_len, step=1)

            # draw frame
            plt.plot(global_time, data_frame)

            # the original start stop line
            start_line = (start, "orange")
            stop_line = (stop, "orange")

            title = "interval nr {}".format(interval_idx)

            fig = plt.gcf()

            # draw lines
            redraw_vlines([start_line, stop_line])

            plt.show(block=False)
            
            def onclick(event):

                global start_line, cid

                plt.clf()
                plt.plot(global_time, data_frame)
                #redraw_vlines([start_line, stop_line])

                plt.show(block=False)

                # get coordinates
                ix = int(event.xdata)

                new_start_line = (ix, "red")
                print("ix", ix)

                # show new start line
                redraw_vlines([new_start_line, stop_line, start_line])
                
                plt.show(block=False)
                
                accept = askyesno(title, "accept start {} ?".format(ix))

                if accept:
                    # set new start
                    start_line = new_start_line
                    print("start line", start_line[0])
                    #fig.canvas.mpl_disconnect(cid)
                    #plt.cla()
                    plt.show(block=False)
                    plt.close()       

            accept = askyesno(title, "accept start {} ?".format(start))

            cid = fig.canvas.mpl_connect('button_press_event', onclick)

            if accept:
                plt.show(block=False)
            else:
                
                plt.show()

            
            # confirm stop line

            #plt.show(block=False)
            
            def onclick(event):

                global stop_line

                plt.clf()
                plt.plot(global_time, data_frame)
                #redraw_vlines([start_line, stop_line])

                plt.show(block=False)

                # get coordinates
                ix = int(event.xdata)

                new_stop_line = (ix, "red")
                print("ix", ix)

                # show new start line
                redraw_vlines([new_stop_line, stop_line, start_line])
                
                plt.show(block=False)
                
                accept = askyesno(title, "accept stop {} ?".format(ix))

                if accept:

                    # set new start
                    stop_line = new_stop_line
                    print("stop line", stop_line[0])
                    #plt.cla()
                    plt.show(block=False)
       

            plt.plot(global_time, data_frame)
            redraw_vlines([start_line, stop_line])
            plt.show(block=False)
            plt.gcf().canvas.mpl_disconnect(cid)

            accept = askyesno(title, "accept stop {} ?".format(stop))
            
            cid = plt.gcf().canvas.mpl_connect('button_press_event', onclick)

            if accept:
                plt.show(block=False)
            else:
                
                plt.show()

            print(start_line)

            m = Meassurement(start_line[0], stop_line[0])
            #m.graph_color = color.get_color()

            self.measurenents.append(m)
            print(str(self.measurenents[0]))

            fig.canvas.mpl_disconnect(cid)

        # TODO handle stop input

        print(self.measurenents)
        #input()

        for messure in self.measurenents:
            messure.calculate(data, self)

        for gr in self.measurenents:
            plt.plot(gr.fs, gr.rts, color=gr.graph_color)

        plt.show(block=False)

        print("done start")



           # draw list
        # 
        graphs = tk.Tk()

        self.show = dict()

        def on_cb_change():

            plt.cla()

            for mes, var in self.show.items():
                if var.get():
                    if not mes.is_calculated():
                        mes.calculate(data, self)

                    plt.plot(mes.fs, mes.rts, color=mes.graph_color)

            plt.ylabel("RT60 (s)")
            plt.xlabel("frquency (Hz)")
            plt.show(block=False)


 
        for id, messure in enumerate(self.measurenents):
            self.show[messure] = tk.BooleanVar(master=graphs, value=True)
            #show += [tk.BooleanVar(master=graphs, value=True)]
            print(messure.start)
            label = tk.Label(master=graphs, text="{}, {}".format(messure.start, messure.stop))
            label.pack()
            color_label = tk.Label(master=graphs, text="COLOR", fg=messure.graph_color, bg=messure.graph_color)
            color_label.pack()
            cb = tk.Checkbutton(master=graphs, variable=self.show[messure], onvalue=True, offvalue=False, command=on_cb_change)
            cb.pack()
            #checkbox = tk.Checkbutton(master=graphs, command=check_box(id), variable=var, onvalue=True, offvalue=False)

        return
            
        #     plt.plot(global_time, data_frame)

        #     coords = []

        #     fig = plt.gcf()
        #     # plot set borders
        #     # TODO show actual updated start border
        #     fig.get_axes()[0].vlines([frame_border, frame_len - frame_border], np.min(data_frame), np.max(data_frame), color=["green"])

        #     plt.show(block=False)

        #     def onclick(event):
        #         global coords, ix, iy, accept

        #         plt.clf()
        #         plt.plot(global_time, data_frame)
        #         plt.show(block=True)

        #         # get coordinates
        #         ix = event.xdata

        #         #coords.append((ix, iy))

        #         #print(coords)

        #         fig.get_axes()[0].vlines([ix], np.min(data), np.max(data), color=["red"])
                
        #         plt.show(block=True)

        #         accept = askyesno("accept stop", "accept stop {} ?".format(ix))

        #         if accept:
        #             print()
        #             plt.close()

        #         #if accept:


        #     cid = fig.canvas.mpl_connect('button_press_event', onclick)

        #     accept = askyesno("accept stop", "accept stop {} ?".format(stop))

        #     #while not accept:
        #     if accept:
        #         upper = stop
                
        #     plt.show()

        #     print("done")

        #     vlines = []
                

        # #fig, ax = plt.subplots(1)
        # #ax = plt.subplot(0)

        # #print(type(ax))
        # plt.show(block=False)

        # #ax.pop().plot(data)
        # #print(askinteger("test","star sample", initialvalue=60000, minvalue=0, maxvalue=len(data)))
        # #ax.pop().axvline(start, 0, 1)

        # #for start, stop in intervals:
        # #    plt.vlines([start, stop], 0, np.max(data), colors=['red'])
            
        # plt.show()

        for mes in self.measurenents:
            pass

        graphs = tk.Tk()

        ints = {}
        for idx, inter in enumerate(intervals):
            ints[idx] = (inter, tk.BooleanVar(master=graphs, value=True))

        def check_box(id: int):
            def f():
                
                plt.cla()

                rts_acc = None

                n_rts = 0

                for key, val in ints.items():

                    _start, _stop = val[0]
                    v: tk.BooleanVar = val[1]

                    if v.get():

                        n_rts += 1
                        fs, rts, _, _ = calc_rt(get_spectogram(data[_start : _stop], {"nfft": self.nfft.get(), "nperseg": self.nperseg.get(), "sr": 48000}), {"thresh": -20})
                        if rts_acc is None:
                            rts_acc = np.zeros(rts.shape)

                        rts_acc += rts
                        
                        plt.plot(fs, rts)
                
                rts_acc /= n_rts
                #plt.plot(fs, rts_acc)
                plt.show(block=False)

            return f

        for id in ints:
            ((start, stop), var) = ints[id]
            #self.plot(data[start : stop])
            label = tk.Label(master=graphs, text="{}, {}".format(start, stop))
            label.pack()
            checkbox = tk.Checkbutton(master=graphs, command=check_box(id), variable=var, onvalue=True, offvalue=False)
            checkbox.pack()
            #fs, rts, _, _ = calc_rt(get_spectogram(data[start : stop], {"nfft": self.nfft.get(), "nperseg": self.nperseg.get(), "sr": 48000}), {"thresh": -20})
            #self.plot(data[start : stop])
            check_box(0)()
            #plt.plot(fs, rts)
            #plt.show(block=False)

            pass

        plt.show()
        return


        


        for idx, (start, stop) in enumerate(intervals):
            if idx == 0 or idx == 1:
                continue
            self.plot(data[stop - int(48000 * 0.5) : stop - 10000])
        


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


        self.fig, self.axis = plt.subplots(1, 4, figsize=(10, 10))
        fig = self.fig
        axis = self.axis
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

        #fs = fs[:300]
        #rts = rts[:300]



        if self.acc_rts is None:
            self.acc_rts = np.zeros(fs.shape)
        self.acc_rts += rts
        self.n_rts += 1

        #plt.scatter(fs, self.acc_rts / self.n_rts, s=5)
        #plt.show()
        print("length", len(fs))

        self.all_rts.append(rts)

        #all_figure, a = plt.subplots()
        #a = plt.subplot(1, 1, 1)

        for graph in self.all_rts:
            axis[2].scatter(fs, graph, s=5)
            #a.scatter(fs, graph, s=5)
        #axis[2].scatter(fs, rts, s=5)
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

def calc_rt(spec: Spectrogram, args: dict)       ->     tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    fs, ts, M = spec.frequencies, spec.times, spec.values

    f_min = 300
    f_max = 6000

    freq_slice = np.where((fs >= f_min) & (fs <= f_max))

    fs = fs[freq_slice]
    M = M[freq_slice,:][0]


    rts = np.zeros(fs.shape)
    mses = np.zeros(fs.shape)
    slopes = np.zeros(fs.shape)
    
    for f_idx, amps in enumerate(M):

        amps_db = to_dB(amps)

        amps_slice = np.array_split(amps_db, 10)

        max_amps = np.array(list(map(np.average, amps_slice)))

        _ts = np.linspace(0, max(ts), num=10)

        #plt.plot(max_amps)
        #plt.show()

        if np.max(amps_db) < args["thresh"]:
            continue

        # linear regression 
        model = np.polyfit(ts, amps_db, 1)
       # model = np.polyfit(_ts, max_amps, 1)

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

        graph_fit = slope * ts + intersect
        #_graph_fit = slope * _ts + intersect

        # if fs[f_idx] > 3080 and fs[f_idx] < 3100:

        #     print("idx", f_idx)
        #     plt.title("frequency {}, slope {}, mse {}".format(fs[f_idx], slope, mses[f_idx]))
        #     plt.plot(amps_db)
        #     plt.plot(graph_fit)
        #     plt.show()

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
        self.ondraw()
        #draw(glob_data)
        

    def on_dec(self):
        if self.mode == "linear":
            self.val.set(self.val.get() - 1)
        else:
            self.val.set(int(self.val.get() / 2))
        self.ondraw()

    def get(self):
        return self.val.get()

        

    def __init__(self, m: tk.Misc, r: int, var, text: str, ondraw, mode: str = "linear") -> None:
        self.val = var
        #self.val.set(256)
        self.mode = mode

        self.ondraw = ondraw

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







