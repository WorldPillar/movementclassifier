import pandas as pd
import numpy as np
import os
import cv2
from threading import Thread
from src import *
import joblib
from config import Config

# model = joblib.load("models/new_model.joblib")
# names = ['Эдик', 'Илья', 'Оля']


def load_model():
    try:
        model = joblib.load(f"{Config.get_model()}")
    except FileNotFoundError:
        model = None
    return model


class Recognition:
    __csv_path = 'csv_data'
    __model = load_model()

    @staticmethod
    def __get_names():
        names = []
        for path in os.listdir(Recognition.__csv_path):
            if os.path.isfile(os.path.join(Recognition.__csv_path, path)):
                names.append(path.split('.')[0])

        names = dict(zip(range(len(names)), names))

        return names

    @staticmethod
    def gait_recognition(frame_collection):
        if Recognition.__model is None:
            return None

        names = Recognition.__get_names()

        crop_region = init_crop_region(720, 1280)
        output_array = []
        for frame_idx in range(len(frame_collection)):
            keypoints_with_scores = run_inference(frame_collection[frame_idx], crop_region,
                                                  crop_size=[input_size, input_size])
            output_array.append(keypoints_with_scores.reshape(51))

        ar = np.array(output_array)
        df = pd.DataFrame(ar)
        predict = Recognition.__model.predict(df)
        unique, counts = np.unique(predict, return_counts=True)
        predict_class = dict(zip(unique, counts))
        print(predict_class)
        # classes = zip(unique, counts)
        #
        result = []
        # for k, v in classes:
        #     result.append(f'{names[k]}: {(v / len(predict)) * 100}%')
        #     print(f'{names[k]}: {(v / len(predict)) * 100}%')

        return predict_class

    @staticmethod
    def update_model():
        Recognition.__model = load_model()

# def gait_recognition(frame_collection, w, h):
#     crop_region = init_crop_region(h, w)
#     output_array = []
#     for frame_idx in range(len(frame_collection)):
#         keypoints_with_scores = run_inference(frame_collection[frame_idx], crop_region,
#                                               crop_size=[input_size, input_size])
#         output_array.append(keypoints_with_scores.reshape(51))
#
#     ar = np.array(output_array)
#     df = pd.DataFrame(ar)
#     predict = model.predict(df)
#     _, counts = np.unique(predict, return_counts=True)
#     predict_class = dict(zip(names, counts))
#
#     for k, v in predict_class.items():
#         print(f'{k}: {(v / len(predict)) * 100}%')
#
#
# def start_capture(w, h):
#     cap = cv2.VideoCapture(0)
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
#
#     if not cap.isOpened():
#         print("Cannot open camera")
#         return
#
#     frame_collection = []
#     counter = 0
#
#     while True:
#         ret, frame = cap.read()
#
#         if not ret:
#             print("Can't receive frame (stream end?). Exiting ...")
#             break
#
#         frame_collection.append(frame)
#         counter += 1
#         if counter == 180:
#             gait_thread = Thread(target=gait_recognition, args=(frame_collection, w, h,))
#             gait_thread.start()
#
#             counter = 0
#             frame_collection = []
#
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#         cv2.imshow('frame', frame)
#
#     cap.release()
#     cv2.destroyAllWindows()
#
#
# if __name__ == '__main__':
#     width, height = 1280, 720
#     start_capture(width, height)
