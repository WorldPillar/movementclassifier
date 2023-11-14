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

for i in range(len(csv)):
    df = pd.read_csv(csv[i])
    df.drop(columns=df.columns[0], axis=1, inplace=True)
    frames.append(df)

    y_class = np.ones(df.shape[0]) * i
    classes.append(y_class)

X = pd.concat(frames)
y = np.hstack(tuple(classes))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf = RandomForestClassifier(max_depth=5, random_state=0)
clf.fit(X_train, y_train)

print(clf.score(X_test, y_test))

joblib.dump(clf, "models/new_model.joblib")
