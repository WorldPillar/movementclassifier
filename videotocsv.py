import os
import pandas as pd
import numpy as np
from src import *
from config import Config


class Frames2CSV:

    @staticmethod
    def __crop(frame_collection: list):
        crop_region = init_crop_region(720, 1280)
        output_array = []
        for frame_idx in range(len(frame_collection)):
            keypoints_with_scores = run_inference(frame_collection[frame_idx], crop_region,
                                                  crop_size=[input_size, input_size])
            output_array.append(keypoints_with_scores.reshape(51))

        ar = np.array(output_array)
        df = pd.DataFrame(ar)
        return df

    @staticmethod
    def save2csv(name: str, frames: list):
        dataframe = Frames2CSV.__crop(frames)

        if not os.path.exists(Config.csv_path):
            os.makedirs(Config.csv_path)
        dataframe.to_csv(f"{Config.csv_path}/{name}.csv")
