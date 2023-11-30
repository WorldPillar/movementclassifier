import tkinter
from threading import Thread
import time

import PIL.Image
import PIL.ImageTk
import cv2

from model import Model
from videotocsv import Frames2CSV
from recognition import Recognition
from config import Config

messages = []


class App:
    def __init__(self, window, window_title: str, video_source: int = 0):
        self.window = window
        self.window.title(window_title)
        self.video_source: int = video_source

        self.entry_name = tkinter.Entry(window, width=20)
        self.entry_name.grid(column=0, row=0)

        self.btn_add = tkinter.Button(window, text="Add user", width=10, command=self.add_user)
        self.btn_add.grid(column=1, row=0)

        self.delete_name = tkinter.Entry(window, width=20)
        self.delete_name.grid(column=0, row=1)

        self.btn_del = tkinter.Button(window, text="Delete user", width=10, command=self.delete_user)
        self.btn_del.grid(column=1, row=1)

        self.textbox = tkinter.Text(window, height=40, width=30)
        self.textbox.grid(column=0, row=2, columnspan=2)

        self.cap = CameraCapture(self.video_source)
        self.photo = None

        self.canvas = tkinter.Canvas(window, width=self.cap.width, height=self.cap.height)
        self.canvas.grid(column=2, row=0, rowspan=4)

        self.deleting: bool = False
        self.saving: bool = False
        self.saving_frames: list = []
        self.user_name: str = ''

        self.recognizing: bool = False
        self.recognize_frames: list = []

        self.delay: int = 15
        self.update()

        self.window.mainloop()

    def update(self):
        ret, frame = self.cap.get_frame()

        if not ret:
            self.window.after(self.delay, self.update)
            return

        if not self.deleting:
            if self.saving and len(self.saving_frames) < 300:
                self.saving_frames.append(frame)
            elif self.saving and len(self.saving_frames) == 300:
                print('start training model')
                self.saving = False
                thread_saving = Thread(target=self.save_user)
                thread_saving.start()
                self.check_thread_train(thread_saving)

            if not self.saving and len(self.recognize_frames) < 150:
                self.recognize_frames.append(frame)
            elif not self.saving and not self.recognizing and len(self.recognize_frames) == 150:
                print('start recognizing')
                thread_recognition = Thread(target=self.recognize)
                thread_recognition.start()
                self.check_thread_recognize(thread_recognition)

        self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
        self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)

        self.window.after(self.delay, self.update)

    def check_thread_recognize(self, thread):
        if thread.is_alive():
            self.window.after(100, lambda: self.check_thread_recognize(thread))
        else:
            self.recognizing = False
            self.print_messages()

    def check_thread_train(self, thread):
        if thread.is_alive():
            self.window.after(100, lambda: self.check_thread_train(thread))
        else:
            self.saving = False
            self.btn_add.config(state=tkinter.NORMAL)
            self.btn_del.config(state=tkinter.NORMAL)
            self.print_messages()

    def print_messages(self):
        while messages:
            self.textbox.insert(tkinter.END, messages.pop(0) + '\n')

    def recognize(self):
        self.recognizing = True
        result = Recognition.gait_recognition(self.recognize_frames)
        if result is None:
            print('Модели ещё нет')
            self.recognize_frames = []
            return

        messages.append(time.strftime('%d.%m.%Y %H:%M:%S', time.localtime(time.time())))
        for res in result:
            messages.append(res)

        self.recognize_frames = []

    def add_user(self):
        if self.saving or self.deleting:
            return

        self.btn_add.config(state=tkinter.DISABLED)
        self.btn_del.config(state=tkinter.DISABLED)
        messages.append('Adding user')
        messages.append(time.strftime('%d.%m.%Y %H:%M:%S', time.localtime(time.time())))
        self.print_messages()

        self.saving_frames = []
        self.user_name = self.entry_name.get()
        self.saving = True
        return

    def save_user(self):
        Controller.add_user(self.user_name, self.saving_frames)
        messages.append(f'User {self.user_name} was add to model')

    def delete_user(self):
        if self.saving or self.deleting:
            return

        self.btn_add.config(state=tkinter.DISABLED)
        self.btn_del.config(state=tkinter.DISABLED)
        messages.append('Deleting user')
        messages.append(time.strftime('%d.%m.%Y %H:%M:%S', time.localtime(time.time())))
        self.print_messages()

        self.user_name = self.delete_name.get()
        self.deleting = True
        thread_deleting = Thread(target=self.deleting_user)
        thread_deleting.start()
        self.check_thread_delete(thread_deleting)

    def deleting_user(self):
        messages.append(Controller.delete_user(self.user_name))

    def check_thread_delete(self, thread):
        if thread.is_alive():
            self.window.after(100, lambda: self.check_thread_delete(thread))
        else:
            self.deleting = False
            self.btn_add.config(state=tkinter.NORMAL)
            self.btn_del.config(state=tkinter.NORMAL)
            self.print_messages()


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
                return ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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

    @staticmethod
    def delete_user(name: str):
        message = 'User not found'
        if Model.delete_user(name):
            message = 'User was deleted'
            Recognition.update_model()
        return message


Config.load_config()
App(tkinter.Tk(), "MovementRecognizer")
