import tkinter as tk

class IntervalRow:

    def __init__(self, master, messure, audio_input_obj, on_delete, on_zoom, on_start_set, on_stop_set) -> None:

        self.frame = tk.Frame(master=master)
        self.frame.pack()

        self.label = tk.Label(master=self.frame, text="{}, {}".format(messure.start, messure.stop))
        #self.label.grid(row=row, column=0)
        self.label.pack(side=tk.LEFT)

        self.btn_remove = tk.Button(master=self.frame, text="delete", command=self.remove)
        #self.btn_remove.grid(row=row, column=1)
        self.btn_remove.pack(side=tk.LEFT)

        self.btn_zoom = tk.Button(master=self.frame, text="zoom in", command=on_zoom(messure))
        self.btn_zoom.pack(side=tk.LEFT)
        
        set_start = on_start_set(messure)

        self.btn_set_start = tk.Button(master=self.frame, text="set start", command=on_start_set(messure))
        self.btn_set_start.pack(side=tk.LEFT)

        self.btn_set_stop = tk.Button(master=self.frame, text="set stop", command=on_stop_set(messure))
        self.btn_set_stop.pack(side=tk.LEFT)


        self.obj = audio_input_obj
        self.messure = messure

        self.on_delete = on_delete

        self.on_zoom = on_zoom

    def remove(self):
        self.obj.measurenents.remove(self.messure)
        print(self.obj.measurenents)

        self.label.pack_forget()
        self.btn_remove.pack_forget()
        self.btn_zoom.pack_forget()

        self.btn_set_start.pack_forget()
        self.btn_set_stop.pack_forget()

        self.on_delete()

    def zoom_in(self):
        self.zoom_in(self.messure)

    def mark(self):
        self.frame.config(background="red")

    def unmark(self):
        self.frame.config(background="white")