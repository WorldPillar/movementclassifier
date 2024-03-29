import os
import numpy as np
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from config import Config

# TODO: Сделать каскад моделей "Один против всех"


class Model:

    @staticmethod
    def __read_csv() -> (list, list):
        csv = []
        names = {}
        i = 0
        for path in os.listdir(Config.csv_path):
            if os.path.isfile(os.path.join(Config.csv_path, path)):
                csv.append(Config.csv_path + '/' + path)
                names[str(i)] = path.split('.')[0]
                i += 1

        Config.set_names(names)
        return csv

    @staticmethod
    def __prepare_x_y(csv: list) -> tuple:
        frames = []
        classes = []

        for pos in range(len(csv)):
            df = pd.read_csv(csv[pos])
            df.drop(columns=df.columns[0], axis=1, inplace=True)

            frames.append(df)

            y_class = np.ones(df.shape[0]) * pos
            classes.append(y_class)

        x = pd.concat(frames).values
        y = np.hstack(tuple(classes))

        return x, y

    @staticmethod
    def train_model():
        csv = Model.__read_csv()

        x, y = Model.__prepare_x_y(csv)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

        clf = RandomForestClassifier(max_depth=5, random_state=0)
        clf.fit(x_train, y_train)

        print(clf.score(x_test, y_test))

        if not os.path.exists(Config.model_path):
            os.makedirs(Config.model_path)
        joblib.dump(clf, f"{Config.get_model()}")
