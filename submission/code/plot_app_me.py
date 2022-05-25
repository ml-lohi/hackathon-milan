import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
import numpy as np
import time
from utils.app_interface import AppInterface
from utils.helper import read_data, to_real
from utils import processing
from scipy import signal
import cv2

# from tensorflow import keras

FOLDER = "hackathon-milan\\submission\\code\\data\\"


def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def calculate_saliency_map(frames):
    new_frames = []
    for frame in frames:
        mean = np.mean(frame)
        frame_new = np.where(frame > mean, frame, 0)
        new_frames.append(frame_new)
    new_frames = np.array(new_frames)
    frames = new_frames
    noise_point = (32, 26)
    r = 3
    # print(frames.shape)
    frames[
        :,
        noise_point[0] - r : noise_point[0] + r,
        noise_point[1] - r : noise_point[1] + r,
    ] = 0
    differences = np.abs(np.diff(frames, axis=0))
    multiplications = []
    for i in range(differences.shape[0] - 1):
        multiplications.append(np.multiply(differences[i], differences[i + 1]))
    multiplications = np.asarray(multiplications)
    multiplications = differences
    # morph = np.array(
    #     [
    #         cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((2, 2)))
    #         for img in multiplications
    #     ]
    # )
    morph = np.array(
        [
            cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((2, 2)))
            for img in multiplications
        ]
    )

    saliency_map_morph = np.expand_dims(np.sum(morph, axis=0), axis=0)
    saliency_map = np.expand_dims(np.sum(multiplications, axis=0), axis=0)

    return normalize_data(saliency_map), normalize_data(saliency_map_morph)


class MatplotlibApp(AppInterface):
    def __init__(self, root=None):
        # self.model = keras.models.load_model(
        #     "hackathon-milan\submission\code\models\CNN"
        # )
        self.data = np.abs(read_data(FOLDER + "3p.csv"))[20:, :, :, :]
        self.root = root
        self.plotFrame = tk.Frame(self.root, bg="black")
        self.plotFrame.pack(side="top", fill="both", expand=True)
        self.buttonFrame = tk.Frame(self.root, bg="black")
        self.buttonFrame.pack(side="bottom", fill="both", expand=True)
        self.plot_active = True
        self.ydata = []
        self.br = 0
        self.strVarBr = tk.StringVar()
        self._reset_hr_br()

    def _reset_hr_br(self):
        self.strVarBr.set(f"Number of people: {self.br}")
        self.root.update()

    def run(self):
        fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
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

        lbl = self.create_label_var(self.buttonFrame, textvariable=self.strVarBr)
        lbl.pack()

    def stop_plotting(self):
        self.plot_active = False

    def start_plotting(self):
        self.plot_active = True

    def restart(self):
        self.ydata = []
        self.br = 0

    def _process(self, canvas, axs):
        data = self.data.reshape(-1, 3, 64, 64)
        for i in range(len(data) - 1):
            sample = data[i : i + 10]
            sample = np.sum(sample, axis=1).squeeze()
            saliency_map = calculate_saliency_map(sample)
            self._plot(canvas, axs, saliency_map[0], saliency_map[1])
            time.sleep(0.5)

    def _plot(self, canvas, axs, saliency_map, saliency_map_morph):

        # to_plot = saliency_map[0, :, :]
        axs[0].imshow(saliency_map[0, :, :])
        axs[1].imshow(saliency_map_morph[0, :, :])

        canvas.draw()


if "__main__" == __name__:
    root = tk.Tk()
    root.title("Main app")
    root.geometry("900x800")
    root.configure(bg="black")
    app = MatplotlibApp(root=root)
    app.run()
    root.mainloop()
