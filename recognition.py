import pandas as pd
import numpy as np
import os
from src import *
import joblib
from config import Config


def load_models():
    models = []
    try:
        for path in os.listdir(Config.model_path):
            if os.path.isfile(os.path.join(Config.model_path, path)):
                models.append(joblib.load(f"{Config.model_path}/{path}"))
    except FileNotFoundError:
        models = None
    return models


class Recognition:
    __models = load_models()

    @staticmethod
    def gait_recognition(frame_collection):
        if Recognition.__models is None:
            return None

        name_position = 0
        result = []
        for model in Recognition.__models:
            crop_region = init_crop_region(720, 1280)
            output_array = []
            for frame_idx in range(len(frame_collection)):
                keypoints_with_scores = run_inference(frame_collection[frame_idx], crop_region,
                                                      crop_size=[input_size, input_size])
                output_array.append(keypoints_with_scores.reshape(51))

            ar = np.array(output_array)
            df = pd.DataFrame(ar)
            predict = model.predict(df)

            unique, counts = np.unique(predict, return_counts=True)
            unique = [int(i) for i in unique]
            if 1 not in unique:
                name_position += 1
                continue

            predict_class = dict(zip(unique, counts))
            print(predict_class)

            total = sum(counts)
            chance = predict_class[1] / total
            if chance > 0.25:
                text = f"{Config.names[str(int(name_position))]}: {round(chance, 4) * 100}%"
                result.append(text)

            name_position += 1
        
        if not result:
            result.append('Никто не обнаружен')

        return result

    @staticmethod
    def update_model():
        Recognition.__models = load_models()
