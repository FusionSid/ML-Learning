import pandas
import random
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from sklearn import tree

def clean_dataset(df):
    assert isinstance(df, pandas.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

music_data = pandas.read_csv("music.csv")
music_data = music_data.drop(columns=['instance_id', 'artist_name', 'track_name', 'obtained_date', "key", "mode", "tempo", "music_genre"])
music_data = clean_dataset(music_data)

def train():
    X = music_data.drop(columns=['popularity'])
    y = music_data["popularity"]

    X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.1)
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    score = accuracy_score(y_test, predictions)

    joblib.dump(model, "music.joblib")
    tree.export_graphviz(model, out_file="music.dot", feature_names=['acousticness', 'danceability', 'duration_ms', 'energy',
       'instrumentalness', 'liveness', 'loudness', 'speechiness', 'valence'], class_names=sorted([str(i) for i in y.unique()]), label="all", rounded=True, filled=True)
    print("Accuracy: {:.0%}".format(score))

train()

model = joblib.load("music.joblib")
number = random.randint(1, 5000)
data = list(music_data.drop(columns=['popularity']).iloc[number])

answer = music_data.iloc[number][0]
prediction = model.predict([data])[0]

correct = prediction == answer

print("Prediction: {0} | Answer: {1} | Correct: {2}".format(prediction, answer, correct))