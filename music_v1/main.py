import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from sklearn import tree

def train():
    music_data = pd.read_csv("music.csv")
    X = music_data.drop(columns=['genre'])
    y = music_data["genre"]
    X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.2)

    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    score = accuracy_score(y_test, predictions)
    print(X.info())

    joblib.dump(model, "music.joblib")
    tree.export_graphviz(model, out_file="music.dot", feature_names=["age", "gender"], class_names=sorted(y.unique()), label="all", rounded=True, filled=True)
    print("Accuracy: {:.0%}".format(score))

def run(age:int, gender:int):
    model = joblib.load("music.joblib")
    prediction = model.predict([[age, gender]])
    gender = "male" if gender == 1 else "female"
    print(f"A {age} year old {gender} would most likely enjoy {prediction[0]} music")

run(32, 1)