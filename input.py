import tkinter as tk

def INCREMENT(x): return x + 1
def DECREMENT(x): return x - 1
def DOUBLE(x): return 2 * x


class ValueInputGroup:
    def __init__(self, root, txt, draw_action, r, start_value="15e-3"):

        self.row = r
        # draw action is called after a button is pressed
        self.draw_action = draw_action
        self.text = tk.Label(root, text=txt, height=1, width=20)
        #self.text.insert(tk.END, txt)
        self.text.grid(row=r, column=0)

        self.inc = tk.Button(
            root, text="+", command=self._get_action(INCREMENT))
        self.inc.grid(row=r, column=1)

        self.dec = tk.Button(
            root, text="-", command=self._get_action(DECREMENT))
        self.dec.grid(row=r, column=2)

        self.value = tk.Text(root, height=1, width=10)

        # mock start value
        self.value.insert(tk.END, start_value)
        self.value.grid(row=r, column=3)

        #self.unit = tk.StringVar(master=root, value="mm")
        #self.unit_menu = tk.OptionMenu(root, self.unit, *UNITS)
        #self.unit_menu.grid(row=r, column=4)



    def __bool__(self):
        return self.getValue() != ""

    def _get_action(self, operation):

        def action():
            val = self.getValue()

            if not val:
                return

            if not "e" in val:
                new_val = float(val)
                new_val = operation(new_val)
                new_val = str(new_val)
            else:
            # exponential notation
                val_list = val.split("e")
                new_val = float(val_list[0])
                new_val = operation(new_val)
                new_val = "{0}e{1}".format(str(new_val), val_list[1])

            self.value.delete(1.0, tk.END)
            self.value.insert(tk.END, new_val)

            self.draw_action()

        return action

    # returns the content of the value field
    def getValue(self) -> str:
        return self.value.get("1.0", tk.END)

    # return the value normalized with the selected unit
    def getNormalizedValue(self) -> str:
        return self.getValue()


    def show(self):
        self.text.grid(row=self.row, column=0)
        self.dec.grid(row=self.row, column=1)
        self.inc.grid(row=self.row, column=2)
        self.value.grid(row=self.row, column=3)

    def hide(self):
        self.text.grid_forget()
        self.dec.grid_forget()
        self.inc.grid_forget()
        self.value.grid_forget()


