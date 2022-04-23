# imports
import pandas
import random
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from sklearn import tree

# clean up the dataset to fix the float issue
def clean_dataset(df):
    assert isinstance(df, pandas.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)


# Load the data set
music_data = pandas.read_csv("music.csv")

# Delete the columns i dont need
music_data = music_data.drop(
    columns=[
        "instance_id",
        "artist_name",
        "track_name",
        "obtained_date",
        "key",
        "mode",
        "tempo",
        "music_genre",
    ]
)

# Run the clean set on this
music_data = clean_dataset(music_data)

# The train function
def train():
    # All the columns except "popularity"
    X = music_data.drop(columns=["popularity"])

    # Only the "popularity" column
    y = music_data["popularity"]

    # Unpack the split data into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.1)

    # Create the model
    model = DecisionTreeClassifier()

    # Train the model
    model.fit(X_train, y_train)
    # Make a prediction
    predictions = model.predict(X_test)
    # Check the accuracy of the model
    score = accuracy_score(y_test, predictions)

    # Save the model file
    joblib.dump(model, "music.joblib")

    # create the tree (The csv is 5000 rows so you have to shorten for this to work)
    # The dot/tree image files i included uses only 100 rows because 5k is way to big
    
    # tree.export_graphviz(
    #     model,
    #     out_file="music.dot",
    #     feature_names=[
    #         "acousticness",
    #         "danceability",
    #         "duration_ms",
    #         "energy",
    #         "instrumentalness",
    #         "liveness",
    #         "loudness",
    #         "speechiness",
    #         "valence",
    #     ],
    #     class_names=sorted([str(i) for i in y.unique()]),
    #     label="all",
    #     rounded=True,
    #     filled=True,
    # )

    # Print the accuracy score of the model
    print("Accuracy: {:.0%}".format(score))


# Comment out this line once you finish training and the music.joblib file is created
train()  # train the model and save it


# Load up the model from the file
model = joblib.load("music.joblib")

# Pick a random row from the csv file
number = random.randint(1, 5000)
data = list(music_data.drop(columns=["popularity"]).iloc[number])

# The popularity for the song
answer = music_data.iloc[number][0]

# Use the model to predict the popularity of the song
prediction = model.predict([data])[0]

# Check if prediction is correct
correct = prediction == answer

# print shit
print(
    "Prediction: {0} | Answer: {1} | Correct: {2}".format(prediction, answer, correct)
)
