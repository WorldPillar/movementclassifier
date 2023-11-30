import os
import numpy as np
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from config import Config


class Model:

    @staticmethod
    def delete_user(name: str) -> bool:
        presence: bool = False
        for path in os.listdir(Config.csv_path):
            if os.path.isfile(os.path.join(Config.csv_path, path)):
                if path.split('.')[0] == name:
                    presence = True
                    os.remove(os.path.join(Config.csv_path, path))
        for path in os.listdir(Config.model_path):
            if os.path.isfile(os.path.join(Config.model_path, path)):
                if path.split('.')[0] == name:
                    presence = True
                    os.remove(os.path.join(Config.model_path, path))
        if name in Config.names.values():
            presence = True

        if presence:
            Model.train_model()
        else:
            return False

        return True

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
    def __prepare_x_y(csv: list):
        models: list[tuple] = []
        df_all: list = [None for _ in range(len(csv))]
        classes: list = [None for _ in range(len(csv))]

        for pos in range(len(csv)):
            df = pd.read_csv(csv[pos])
            df.drop(columns=df.columns[0], axis=1, inplace=True)

            for i in range(len(classes)):
                if classes[i] is None:
                    if pos == i:
                        classes[i] = np.ones(df.shape[0])
                    else:
                        classes[i] = np.zeros(df.shape[0])
                    continue
                if pos != i:
                    classes[i] = np.hstack(tuple([classes[i], np.zeros(df.shape[0])]))
                else:
                    classes[i] = np.hstack(tuple([np.ones(df.shape[0]), classes[i]]))

            for i in range(len(df_all)):
                if df_all[i] is None:
                    df_all[i] = df
                    continue
                if pos == i:
                    df_all[i] = pd.concat([df, df_all[i]])
                else:
                    df_all[i] = pd.concat([df_all[i], df])

        for pos in range(len(df_all)):
            models.append((df_all[pos].values, classes[pos]))

        return models

    @staticmethod
    def train_model():
        csv = Model.__read_csv()

        name_pos = 0
        for x, y in Model.__prepare_x_y(csv):
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

            clf = RandomForestClassifier(max_depth=5, random_state=0)
            clf.fit(x_train, y_train)

            print(clf.score(x_test, y_test))

            if not os.path.exists(Config.model_path):
                os.makedirs(Config.model_path)
            joblib.dump(clf, f"{Config.model_path}/{Config.names[str(name_pos)]}.joblib")
            name_pos += 1
