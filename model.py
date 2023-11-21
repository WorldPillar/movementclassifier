import os
import numpy as np
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

frames = []
classes = []

dir_path = "csv_data"

csv = []
for path in os.listdir(dir_path):
    if os.path.isfile(os.path.join(dir_path, path)):
        csv.append(dir_path + '/' + path)

csv_position = 0

while csv_position < len(csv):
    df_1 = pd.read_csv(csv[csv_position])
    df_1.drop(columns=df_1.columns[0], axis=1, inplace=True)

    csv_position += 1

    df_2 = pd.read_csv(csv[csv_position])
    df_2.drop(columns=df_2.columns[0], axis=1, inplace=True)

    df = pd.concat([df_1, df_2])
    frames.append(df)

    y_class = np.ones(df.shape[0]) * ((csv_position - 1) // 2)
    classes.append(y_class)

    csv_position += 1

X = pd.concat(frames)
y = np.hstack(tuple(classes))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf = RandomForestClassifier(max_depth=5, random_state=0)
clf.fit(X_train, y_train)

print(clf.score(X_test, y_test))

joblib.dump(clf, "models/new_model.joblib")
