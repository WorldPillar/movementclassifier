import pandas as pd
import numpy as np
import cv2
from threading import Thread
from src import *
import joblib

model = joblib.load("models/")


def gait_recognition(frame_collection, w, h):
    crop_region = init_crop_region(h, w)
    output_array = []
    for frame_idx in range(len(frame_collection)):
        keypoints_with_scores = run_inference(frame_collection[frame_idx], crop_region,
                                              crop_size=[input_size, input_size])
        output_array.append(keypoints_with_scores.reshape(51))

    ar = np.array(output_array)
    df = pd.DataFrame(ar)
    predict = model.predict(df)
    print("name: " + str(predict) + "%")


# todo: создать список имён файлов аналагично csv для вывода имён при предикте

def start_capture(w, h):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    if not cap.isOpened():
        print("Cannot open camera")
        return

    frame_collection = []
    counter = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        frame_collection.append(frame)
        counter += 1
        if counter == 180:
            gait_thread = Thread(target=gait_recognition, args=(frame_collection, w, h,))
            gait_thread.start()

            counter = 0
            frame_collection = []

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        cv2.imshow('frame', frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    width, height = 1280, 720
    start_capture(width, height)
