import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
import numpy as np
import time
from utils.app_interface import AppInterface
from utils.helper import read_data
from utils import processing
from scipy import signal
import cv2

from tensorflow import keras

FOLDER = "hackathon-milan\\submission\\code\\data\\"


def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def calculate_saliency_map_3(frames, to_morph=True):
    """n_frames, n_radars, 64, 64"""
    """new_frames = []
    for frame in frames:
        mean = np.mean(frame)
        frame_new = np.where(frame > mean, frame, 0)
        new_frames.append(frame_new)
    new_frames = np.array(new_frames)
    frames = new_frames"""
    frames[:, :, 31:34, :] = 0
    # differences = np.abs(np.diff(frames, axis=0))
    # print(differences.shape)
    multiplications = np.zeros((frames.shape[0] - 1, frames.shape[1], 64, 64))
    for i in range(frames.shape[0] - 1):
        for j in range(frames.shape[1]):
            mult = np.multiply(frames[i, j, :, :], frames[i + 1, j, :, :])
            multiplications[i, j, :, :] = mult

    if to_morph:
        morphology = np.zeros(multiplications.shape)
        for i in range(multiplications.shape[0]):
            for j in range(multiplications.shape[1]):
                morph = cv2.morphologyEx(
                    multiplications[i, j, :, :], cv2.MORPH_OPEN, np.ones((2, 2))
                )
                morphology[i, j, :, :] = morph
        saliency_map = np.expand_dims(np.sum(morphology, axis=0), axis=0)
    else:
        saliency_map = np.expand_dims(np.sum(multiplications, axis=0), axis=0)

    return normalize_data(saliency_map).squeeze()


def gaussian_filter(array, sigma=1):
    kernel = cv2.getGaussianKernel(5, sigma)
    # kernel = np.transpose(kernel)
    return cv2.filter2D(array, -1, kernel)


def create_hist(frames):
    frames[:, 31:34, :] = 0
    sum_frames = np.sum(frames, axis=0)
    # sum_frames = np.expand_dims(sum_frames, axis=0)
    hist1 = gaussian_filter(np.sum(sum_frames, axis=1))
    hist2 = gaussian_filter(np.sum(sum_frames, axis=1))
    hist1 = np.expand_dims(hist1, axis=0)
    hist2 = np.expand_dims(hist2, axis=0)
    return np.concatenate((hist1, hist2), axis=0)


def calculate_saliency_map_1(frames):
    new_frames = []
    for frame in frames:
        mean = np.mean(frame)
        frame_new = np.where(frame > mean, frame, 0)
        new_frames.append(frame_new)
    new_frames = np.array(new_frames)
    frames = new_frames
    frames[:, 31:34, :] = 0
    multiplications = []
    for i in range(frames.shape[0] - 1):
        multiplications.append(np.multiply(frames[i], frames[i + 1]))

    multiplications = np.array(multiplications)
    # saliency_map_morph = np.expand_dims(np.sum(differences, axis=0), axis=0)
    saliency_map = np.expand_dims(np.sum(multiplications, axis=0), axis=0)

    return normalize_data(saliency_map).squeeze()


def easy_preprocessing(frames):
    frames[:, :, 31:34, :] = 0
    img = np.sum(frames, axis=0).squeeze()
    img_new = np.zeros(img.shape)
    return normalize_data(img)


class MatplotlibApp(AppInterface):
    def __init__(self, root=None):
        self.model = keras.models.load_model(
            "hackathon-milan\submission\code\models\CNN"
        )
        self.data = np.abs(read_data(FOLDER + "2p.csv"))[:, :, :, :]
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
        fig, axs = plt.subplots(1, 1, figsize=(10, 5), sharex=True, sharey=True)
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
        true_label = 2
        i = 0
        counter_false = 0
        counter = 0
        while True:
            sample = np.abs(data[i : i + 20])
            sample = easy_preprocessing(sample)
            sample = np.moveaxis(sample, 0, 2)
            predicted = self.model.predict(np.expand_dims(sample, axis=0))
            print(predicted)
            label = np.argmax(predicted)
            counter = counter + 1
            if label != true_label:
                counter_false += 1
            print(f"counter: {counter}, counter_false: {counter_false}")
            self.br = label
            self._reset_hr_br()

            # self._plot(canvas, axs, saliency_map)
            time.sleep(0.25)
            i += 5
            if i + 20 >= len(data):
                break

    def _plot(self, canvas, axs, saliency_map):
        axs.cla()
        axs.imshow(saliency_map[:, :])

        canvas.draw()


if "__main__" == __name__:
    root = tk.Tk()
    root.title("Main app")
    root.geometry("900x800")
    root.configure(bg="black")
    app = MatplotlibApp(root=root)
    app.run()
    root.mainloop()
