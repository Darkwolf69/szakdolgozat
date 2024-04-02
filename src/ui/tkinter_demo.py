# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 01:30:59 2024

@author: adamWolf
"""

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askopenfilename


LARGE_FONT = ("Verdana", 12)
MEDIUM_FONT = ("Verdana", 10)
SMALL_FONT = ("Verdana", 10)
MY_TEXT = ''
    

class NLP_app(tk.Tk):

    def __init__(self, *args, **kwargs):
        
        tk.Tk.__init__(self, *args, **kwargs)

        # tk.Tk.iconbitmap(self)
        tk.Tk.wm_title(self, "NLP app")
        
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand = True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        

        self.frames = {}

        for F in (StartPage, Basic_stats_page, Plot1_page, Plot2_page, Plot3_page):
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")
            
        self.show_frame(StartPage)

    def show_frame(self, cont):

        frame = self.frames[cont]
        frame.tkraise()

        
class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self,parent)
        
        self.label2_var=tk.StringVar()
        
        label = tk.Label(self, text="Natural Language Processing application", font=LARGE_FONT)
        label.pack(pady=20,padx=20)
        
        self.label2 = tk.Label(self, textvariable=self.label2_var, font=MEDIUM_FONT)
        self.label2.place(x=25, y=50)
        self.label2.pack(pady=0,padx=0)


        open_file_button = ttk.Button(self, text="Open file",
                            command=lambda: self.open_file())
        open_file_button.pack()
        open_file_button.place(x=25, y=100)

        
        start_page_basic_stats_btn = ttk.Button(self, text="Basic statistics",
                            command=lambda: [controller.show_frame(Basic_stats_page), Basic_stats_page.set_text(MY_TEXT)])
        start_page_basic_stats_btn.pack()
        start_page_basic_stats_btn.place(x=25, y=125)


        start_page_plt1_btn = ttk.Button(self, text="Graph Page 1",
                            command=lambda: controller.show_frame(Plot1_page))
        start_page_plt1_btn.pack()
        start_page_plt1_btn.place(x=25, y=150)


        start_page_plt2_btn = ttk.Button(self, text="Graph Page 2",
                            command=lambda: controller.show_frame(Plot2_page))
        start_page_plt2_btn.pack()
        start_page_plt2_btn.place(x=25, y=175)


        start_page_plt3_btn = ttk.Button(self, text="Graph Page 3",
                            command=lambda: controller.show_frame(Plot3_page))
        start_page_plt3_btn.pack()
        start_page_plt3_btn.place(x=25, y=200)
    
    
    def open_file(self):
        filepath = askopenfilename(
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        if not filepath:
            return
        with open(filepath, mode="r", encoding="utf-8") as input_file:
            MY_TEXT = input_file.read()
            print(f"MY_TEXT: {MY_TEXT}")
        
        self.label2.config(text = f"Actual filepath: - {filepath}")
        
        
    def hide_label(self): 
        self.label2.pack_forget() 
  
    
    def show_label(self): 
        self.label2.pack()


class Basic_stats_page(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Basic statistics of text", font=LARGE_FONT)
        label.pack(pady=10,padx=10)
        self.label2 = tk.Label(self, text = MY_TEXT, font=LARGE_FONT)
        self.label2.place(x=25, y=50)
        self.label2.pack(pady=50,padx=50)

        basic_stats_page_start_btn = ttk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage))
        basic_stats_page_start_btn.pack()
        
    def set_text(self):
        self.label2.config(text = MY_TEXT)


class Plot1_page(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Graph Page 1", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        plt1_page_start_btn = ttk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage))
        plt1_page_start_btn.pack()

        plt1_page_plt2_btn = ttk.Button(self, text="Graph Page 2",
                            command=lambda: controller.show_frame(Plot2_page))
        plt1_page_plt2_btn.pack()
        
        plt1_page_plt3_btn = ttk.Button(self, text="Graph Page 3",
                            command=lambda: controller.show_frame(Plot3_page))
        plt1_page_plt3_btn.pack()


class Plot2_page(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Graph Page 2", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        plt2_page_start_btn = ttk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage))
        plt2_page_start_btn.pack()

        plt2_page_plt1_btn = ttk.Button(self, text="Graph Page 1",
                            command=lambda: controller.show_frame(Plot1_page))
        plt2_page_plt1_btn.pack()
        
        plt2_page_plt3_btn = ttk.Button(self, text="Graph Page 3",
                            command=lambda: controller.show_frame(Plot3_page))
        plt2_page_plt3_btn.pack()


class Plot3_page(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Graph Page 3", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        plt3_page_start_btn = ttk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage))
        plt3_page_start_btn.pack()
        
        plt3_page_plt1_btn = ttk.Button(self, text="Graph Page 1",
                            command=lambda: controller.show_frame(Plot1_page))
        plt3_page_plt1_btn.pack()
        
        plt3_page_plt2_btn = ttk.Button(self, text="Graph Page 2",
                            command=lambda: controller.show_frame(Plot2_page))
        plt3_page_plt2_btn.pack()


        f = Figure(figsize=(5,5), dpi=100)
        a = f.add_subplot(111)
        a.plot([1,2,3,4,5,6,7,8],[5,6,1,3,8,9,3,5])

        canvas = FigureCanvasTkAgg(f, self)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, self)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        
        
app = NLP_app()
app.mainloop()
        