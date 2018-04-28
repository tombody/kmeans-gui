from tkinter import *
import tkinter.messagebox

TITLE = "Thomas Body - U1101544"

class Interface:
    def __inti__(self, master):
        # Sets root title and min window size
        self.master = master
        master.title(TITLE)
        master.minsize(width=500, height=40)