from tkinter import *
import tkinter.messagebox

TITLE = "K-Means - Thomas Body - U1101544"


class Interface:
    def __init__(self, master):
        # Sets root title and min window size
        self.master = master
        master.title(TITLE)
        master.minsize(width=500, height=40)


# Runs the main root display
if __name__ == "__main__":
    root = Tk()
    interface = Interface(root)
    root.mainloop()
