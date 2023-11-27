import tkinter
from threading import Thread

import PIL.Image
import PIL.ImageTk
import cv2

from model import Model
from videotocsv import Frames2CSV
from recognition import Recognition
from config import Config


class App:
    def __init__(self, window, window_title: str, video_source: int = 0):
        self.window = window
        self.window.title(window_title)
        self.video_source: int = video_source

        self.entry_name = tkinter.Entry(window, width=30)
        self.entry_name.pack(anchor=tkinter.NW, expand=True)

        self.btn_add = tkinter.Button(window, text="Add user", width=20, command=self.add_user)
        self.btn_add.pack(anchor=tkinter.NW, expand=True)

        self.cap = CameraCapture(self.video_source)
        self.photo = None

        self.canvas = tkinter.Canvas(window, width=self.cap.width, height=self.cap.height)
        self.canvas.pack()

        self.saving: bool = False
        self.saving_frames: list = []
        self.user_name: str = ''

        self.recognize_frames: list = []
        self.labels = []

        self.delay: int = 15
        self.update()

        self.window.mainloop()

    def update(self):
        ret, frame = self.cap.get_frame()

        if not ret:
            self.window.after(self.delay, self.update)

        if self.saving and ret and len(self.saving_frames) < 300:
            self.saving_frames.append(frame)
        elif self.saving and ret and len(self.saving_frames) == 300:
            thread = Thread(target=self.save_user())
            thread.start()

        if not self.saving and len(self.recognize_frames) < 150:
            self.recognize_frames.append(frame)
        elif not self.saving and len(self.recognize_frames) == 150:
            thread = Thread(target=self.recognize())
            thread.start()

        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)

        self.window.after(self.delay, self.update)

    def recognize(self):
        result = Recognition.gait_recognition(self.recognize_frames)
        if result is None:
            print('Модели ещё нет')
        self.update_labels(result)

        self.recognize_frames = []

    def update_labels(self, result):
        for label in self.labels:
            label.destroy()

        self.labels = []
        res_sum = sum(result.values())

        for res in result:
            text = f"{Config.names[str(int(res))]}: {round(result[res] / res_sum, 2) * 100}%"
            self.labels.append(tkinter.Label(self.window,
                                             text=text, width=20))
            self.labels[-1].pack(anchor=tkinter.NW, expand=True)

    def add_user(self):
        if self.saving:
            return

        self.saving_frames = []
        self.user_name = self.entry_name.get()
        self.saving = True

        print(self.user_name)
        return

    def save_user(self):
        self.saving = False
        Controller.add_user(self.user_name, self.saving_frames)

# TODO: Сделать постоянное распознавание (каждый 150 кадров)
# TODO: Добавить вывод ошибок, проверки условий


class CameraCapture:
    def __init__(self, video_source=0):
        self.cap = cv2.VideoCapture(video_source)
        if not self.cap.isOpened():
            raise ValueError("Unable to open video source", video_source)

        self.width = 1280
        self.height = 720

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def get_frame(self):
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                return ret, frame
            else:
                return ret, None

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()


class Controller:
    @staticmethod
    def add_user(name: str, frames: list):
        Frames2CSV.save2csv(name, frames)
        Model.train_model()
        Recognition.update_model()


Config.load_config()
App(tkinter.Tk(), "MovementRecognizer")
