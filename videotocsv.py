import os
import pandas as pd
import numpy as np
import cv2
from src import *


def crop(frame_collection: list):
    crop_region = init_crop_region(720, 1280)
    output_array = []
    for frame_idx in range(len(frame_collection)):
        keypoints_with_scores = run_inference(frame_collection[frame_idx], crop_region,
                                              crop_size=[input_size, input_size])
        output_array.append(keypoints_with_scores.reshape(51))

    ar = np.array(output_array)
    df = pd.DataFrame(ar)
    return df


def start_capture(w, h) -> (bool, list):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    if not cap.isOpened():
        print("Cannot open camera")
        return False, []

    frame_collection = []
    counter = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            return False, []

        frame_collection.append(frame)
        counter += 1
        if counter == 300:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        cv2.imshow('frame', frame)

    print(cap.get(3), cap.get(4))
    cap.release()
    cv2.destroyAllWindows()
    return True, frame_collection


def video_save(file_name, frame_collection: list, w, h):
    vid_path = 'videos'
    if not os.path.exists(vid_path):
        os.makedirs(vid_path)
    
    result = cv2.VideoWriter(f'{vid_path}/{file_name}.mp4',
                             cv2.VideoWriter_fourcc(*'mp4v'),
                             30.0, (w, h))
    for frame in frame_collection:
        result.write(frame)
    result.release()


if __name__ == '__main__':
    width, height = 1280, 720
    is_ret, frames = start_capture(width, height)
    if not is_ret:
        print("Something went wrong")
        exit()

    dataframe = crop(frames)

    name = 'ilia'

    csv_path = 'cvs_data'
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)
    dataframe.to_csv(f"{csv_path}/{name}.csv")

    video_save(name, frames, width, height)
