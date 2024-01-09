import cv2
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np

import code.findTarget

class main:
    def __init__(self):
        self.video_source = 0
        self.vid = cv2.VideoCapture(self.video_source)
        self.root = tk.Tk()
        self.root.title("Object Detection")
        self.canvas = tk.Canvas(self.root, width=self.vid.get(3), height=self.vid.get(4))
        self.canvas.pack()

        self.start_button = tk.Button(self.root, text="Start Stream", command=self.start_stream)
        self.start_button.pack(side=tk.LEFT, padx=5)

        self.triangle_button = tk.Button(self.root, text="Find Triangle", command=self.find_triangle)
        self.triangle_button.pack(side=tk.LEFT, padx=5)

        self.rectangle_button = tk.Button(self.root, text="Find Rectangle", command=self.find_rectangle)
        self.rectangle_button.pack(side=tk.LEFT, padx=5)

        self.toggle_button = tk.Button(self.root, text="Toggle Objects", command=self.toggle_objects)
        self.toggle_button.pack(side=tk.LEFT, padx=5)

        self.is_streaming = False
        self.is_triangle_enabled = False
        self.is_circle_enabled = False

        self.update()
        self.root.mainloop()

    def start_stream(self):
        self.is_streaming = not self.is_streaming
        if self.is_streaming:
            self.start_button["text"] = "Stop Stream"
        else:
            self.start_button["text"] = "Start Stream"

    def find_triangle(self):
        if not self.is_streaming: return
        self.is_triangle_enabled = not self.is_triangle_enabled

    def find_rectangle(self):
        if not self.is_streaming: return
        self.is_circle_enabled = not self.is_circle_enabled

    def toggle_objects(self):
        if not self.is_streaming: return
        if self.is_circle_enabled or self.is_triangle_enabled:
            self.is_triangle_enabled = True
            self.is_circle_enabled = True
        else:
            self.is_triangle_enabled = True
            self.is_circle_enabled = True


    def detect_objects(self, frame):
        if self.is_triangle_enabled:
            code.findTarget.findAndDrawTriangle(frame)
        if self.is_circle_enabled:
            code.findTarget.findAndDrawCircle(frame)

    def update(self):
        if self.is_streaming:
            ret, frame = self.vid.read()
            if ret:
                self.detect_objects(frame)
                self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
            else:
                print("error")
        self.root.after(100, self.update)

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
