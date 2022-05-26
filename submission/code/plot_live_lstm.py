from tabnanny import verbose
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
import numpy as np
import time
from utils.app_interface import AppInterface
from ifxdaq.sensor.radar_ifx import RadarIfxAvian
from utils import processing
from tensorflow import keras


class MatplotlibApp(AppInterface):
    def __init__(self, root=None):
        self.root = root
        self.model = keras.models.load_model(
            "hackathon-milan/submission/code/models_kosta/LSTM"
        )
        self.plotFrame = tk.Frame(self.root, bg="black")
        self.plotFrame.pack(side="top", fill="both", expand=True)
        self.buttonFrame = tk.Frame(self.root, bg="black")
        self.buttonFrame.pack(side="bottom", fill="both", expand=True)
        self.plot_active = True
        self.raw_data = []
        self.br = 0
        self.number_of_frames = 5
        self.strVarBr = tk.StringVar()
        self._reset_hr_br()
        self.config_file = (
            "hackathon-milan\\examples\\radar_configs\\RadarIfxBGT60.json"
        )
        # self.flag = False

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
        # self.button_start = self.create_button(self.buttonFrame, text="Restart")
        # self.button_start.bind("<Button-1>", lambda event: self.restart())
        # self.button_start.pack()

        lbl = self.create_label_var(self.buttonFrame, textvariable=self.strVarBr)
        lbl.pack()

    def stop_plotting(self):
        self.plot_active = False

    def start_plotting(self):
        self.plot_active = True
        self.raw_data = []

    def restart(self):
        self.row_data = []
        self.br = 0

    def _process(self, canvas, axs):

        with RadarIfxAvian(
            self.config_file
        ) as device:  # Initialize the radar with configurations
            for i_frame, frame in enumerate(
                device
            ):  # Loop through the frames coming from the radar
                if self.plot_active:
                    if i_frame != 0:
                        self.raw_data.append(
                            np.squeeze(frame["radar"].data / (4095.0))
                        )  # Dividing by 4095.0 to scale the data
                    if i_frame % (self.number_of_frames) == 0 and i_frame != 0:
                        data = np.array(self.raw_data)
                        range_doppler_map = processing.processing_rangeDopplerData(data)
                        sample = np.abs(np.expand_dims(range_doppler_map, axis=0))
                        sample[:, :, :, 32, :] = 0
                        sample = np.moveaxis(sample, 2, 4)
                        prediction_array = self.model.predict(sample, verbose=0)
                        label = np.argmax(prediction_array)
                        self.br = label
                        self._reset_hr_br()
                        self._plot(canvas, axs, sample.squeeze())
                        self.raw_data = []

    def _plot(self, canvas, axs, sample):

        axs.cla()
        sample = np.sum(sample, axis=3).squeeze()
        sample = np.sum(sample, axis=0).squeeze()
        to_plot = sample
        axs.imshow(to_plot)

        canvas.draw()
        # print(f"Plotted")


if "__main__" == __name__:
    root = tk.Tk()
    root.title("Main app")
    root.geometry("900x800")
    root.configure(bg="black")
    app = MatplotlibApp(root=root)
    app.run()
    root.mainloop()
