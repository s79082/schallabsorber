import tkinter as tk

class IntervalRow:

    def __init__(self, master, messure, audio_input_obj, on_delete) -> None:
        self.label = tk.Label(master=master, text="{}, {}".format(messure.start, messure.stop))
        #self.label.grid(row=row, column=0)
        self.label.pack()

        self.btn_remove = tk.Button(master=master, text="delete", command=self.remove)
        #self.btn_remove.grid(row=row, column=1)
        self.btn_remove.pack()

        self.obj = audio_input_obj
        self.messure = messure

        self.on_delete = on_delete

    def remove(self):
        self.obj.measurenents.remove(self.messure)
        print(self.obj.measurenents)

        self.label.pack_forget()
        self.btn_remove.pack_forget()

        self.on_delete()