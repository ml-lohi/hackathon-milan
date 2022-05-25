import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
import numpy as np
import time
from utils.app_interface import AppInterface
from utils.helper import read_data
from utils import processing

FOLDER = "hackathon-milan\\submission\\code\\data\\"


class MatplotlibApp(AppInterface):
    def __init__(self, root=None):
        self.data = read_data(FOLDER + "3p.csv")
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
        fig, axs = plt.subplots(3, 5, figsize=(10, 5), sharex=True, sharey=True)
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
        # for ax in axs:
        #     ax.set_facecolor("xkcd:black")
        #     color = "white"
        #     ax.xaxis.label.set_color(color)
        #     ax.yaxis.label.set_color(color)
        #     ax.tick_params(axis="x", colors=color)
        #     ax.tick_params(axis="y", colors=color)

        for sample in self.data:
            sample = np.squeeze(sample)
            self._plot(canvas, axs, sample)
            self.br = self.br + 1
            time.sleep(0.25)

    def _plot(self, canvas, axs, range_doppler_map):
        for i in range(3):
            for j in range(5):
                axs[i, j].imshow(np.abs(range_doppler_map)[j, i, :, :])
                axs[i, j].set_aspect("equal")

        canvas.draw()


if "__main__" == __name__:
    root = tk.Tk()
    root.title("Main app")
    root.geometry("900x800")
    root.configure(bg="black")
    app = MatplotlibApp(root=root)
    app.run()
    root.mainloop()
