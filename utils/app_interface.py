from abc import abstractmethod
from typing import Protocol
import tkinter as tk


class AppInterface(Protocol):
    
    def create_label(self, frame, text:str = "Label"):
        label = tk.Label(
            frame,
            text=text,
            foreground="green",  
            background="black"  
        )
        label.config(font=('Helvatical bold',20))
        return label

    def create_label_var(self, frame, textvariable:tk.StringVar):
        label = tk.Label(
            frame,
            textvariable=textvariable,
            foreground="green",  
            background="black"
        )
        label.config(font=('Helvatical bold',20))
        return label

    def create_button(self, frame:tk.Frame, text:str = "Button"):
        button = tk.Button(
            frame,
            text=text,
            width=10,
            bg="black",
            fg="green",
            highlightbackground="black", # Fix: for MAC https://stackoverflow.com/questions/1529847/how-to-change-the-foreground-or-background-colour-of-a-tkinter-button-on-mac-os
        )
        return button
    
    def create_entry(self, frame:tk.Frame):
        entry = tk.Entry(
            frame,
            foreground="green",
            background="black"
        )
        entry.config(font=('Helvatical bold',20))
        return entry
    
    def clear_frame(self, frame):
        for widgets in frame.winfo_children():
            widgets.destroy()
    
    @abstractmethod
    def run(self):
        pass
    

