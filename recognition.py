import pandas as pd
import numpy as np
import os
from src import *
import joblib
from config import Config


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

        total = sum(counts)
        result = []
        for k, v in predict_class.items():
            text = f"{Config.names[str(int(k))]}: {round(v / total, 4) * 100}%"
            result.append(text)

        return result

    @staticmethod
    def update_model():
        Recognition.__model = load_model()
