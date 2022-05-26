import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
import numpy as np
import time
from utils.app_interface import AppInterface
from utils.helper import read_data, morphology
from utils import processing
import cv2
from tensorflow import keras

FOLDER = "hackathon-milan\\submission\\code\\data\\"
PATH_MAC_MODEL = "submission/code/models/LSTM_final"
PATH_MAC_DATA = "submission/code/data/"


class MatplotlibApp(AppInterface):
    def __init__(self, root=None):
        self.model = keras.models.load_model(
            PATH_MAC_MODEL
        )
        self.data = read_data(PATH_MAC_DATA + "2p.csv", n_frames = 20)
        self.root = root
        self.plotFrame = tk.Frame(self.root, bg="black")
        self.plotFrame.pack(side="top", fill="both", expand=True)
        self.buttonFrame = tk.Frame(self.root, bg="black")
        self.buttonFrame.pack(side="bottom", fill="both", expand=True)
        self.plot_active = True
        self.ydata = []
        self.people_count = 0
        self.strVarPeopleCount = tk.StringVar()
        self._reset_var()

    def _reset_var(self):
        self.strVarPeopleCount.set(f"Number of people: {self.people_count}")
        self.root.update()

    def run(self):
        fig, axs = plt.subplots(1, 3, figsize=(3, 3), sharex=True, sharey=True)
        fig.suptitle("Range-Doppler Plot")
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.plotFrame)
        canvas.get_tk_widget().pack()
        canvas.draw()
        self.thread = threading.Thread(target=self._process, args=(canvas, axs))
        self.thread.setDaemon(True)
        self.thread.start()

        self.button_stop = self.create_button(self.buttonFrame, text="Stop")
        self.button_stop.bind("<Button-1>", lambda event: self.stop_plotting())
        self.button_stop.pack()
        self.button_start = self.create_button(self.buttonFrame, text="Start")
        self.button_start.bind("<Button-1>", lambda event: self.start_plotting())
        self.button_start.pack()
        self.button_start = self.create_button(self.buttonFrame, text="Restart")
        self.button_start.bind("<Button-1>", lambda event: self.restart())
        self.button_start.pack()

        lbl = self.create_label_var(self.buttonFrame, textvariable=self.strVarPeopleCount)
        lbl.pack()

    def stop_plotting(self):
        self.plot_active = False

    def start_plotting(self):
        self.plot_active = True

    def restart(self):
        self.ydata = []
        self.people_count = 0

    def _process(self, canvas, axs):
        for sample in self.data:
            sample = np.abs(np.expand_dims(sample, axis=0))
            # sample[:,:,32,:] = 0
            sample[:,:,:,32,:] = 0
            # sample = morphology(sample)
            # sample = np.moveaxis(sample, 1, 3)
            sample = np.moveaxis(sample, 2, 4)
            prediction_array = self.model.predict(sample)
            self._plot(canvas, axs, sample)
            self.people_count = np.argmax(prediction_array)
            self._reset_var()
            time.sleep(0.25)

    def _plot(self, canvas, axs, sample):
        for i, ax in enumerate(axs):
            ax.cla()
            to_plot = sample[0,0,:,:,i]
            ax.imshow(to_plot)
        canvas.draw()


if "__main__" == __name__:
    root = tk.Tk()
    root.title("Main app")
    root.geometry("900x800")
    root.configure(bg="black")
    app = MatplotlibApp(root=root)
    app.run()
    root.mainloop()
