import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
import numpy as np
import time
from utils.app_interface import AppInterface
from utils.helper import read_data, to_real, morphology
from utils import processing
import cv2
from tensorflow import keras

FOLDER = "hackathon-milan\\submission\\code\\data\\data_big_2\\"
PATH_MAC_MODEL = "submission/code/models/CNN"
PATH_MAC_DATA = "submission/code/data/data_big_2/"

def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def calculate_saliency_map(frames):
    noise_point = (32, 26)
    r = 3
    # print(frames.shape)
    frames[
        :,
        noise_point[0] - r : noise_point[0] + r,
        noise_point[1] - r : noise_point[1] + r,
    ] = 0
    differences = np.diff(frames, axis=0)
    # multiplications = []
    # for i in range(differences.shape[0] - 1):
    #     multiplications.append(np.multiply(differences[i], differences[i + 1]))
    # multiplications = np.asarray(multiplications)
    # morph = np.array(
    #     [cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((5, 5))) for img in differences]
    # )
    saliency_map = np.expand_dims(np.sum(differences, axis=0), axis=0)
    return normalize_data(saliency_map)


class MatplotlibApp(AppInterface):
    def __init__(self, root=None):
        # self.model = keras.models.load_model(
        #     PATH_MAC_MODEL
        # )
        self.data = read_data(PATH_MAC_DATA + "2p.csv")
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
        fig, axs = plt.subplots(1, 1, figsize=(3, 3), sharex=True, sharey=True)
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
            # Take mean from 5 samples and summ data from 3 sensors + erase Line
            sample = to_real(np.sum(np.expand_dims(sample, axis=0), axis=1))

            # sample[:,:,32,:] = 0
            # sample = morphology(np.abs(np.expand_dims(np.mean(sample, axis=0), axis=0)))
            # sample = np.expand_dims(np.sum(sample > 1, axis=1), axis=1)

            sample = np.moveaxis(sample, 1, 3)      # [frame, img(64,64), sensor]
            # prediction_array = self.model.predict(sample)
            self._plot(canvas, axs, sample)
            # self.people_count = np.argmax(prediction_array)
            self._reset_var()
            time.sleep(0.25)

    def _plot(self, canvas, axs, sample):
        for i, ax in enumerate(axs):
            to_plot = sample[0,:,:,i]
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
