import pandas as pd, numpy
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import joblib

# Importing the dataset
data = pd.read_csv("Iris.csv")

# Assigning Integers to Labels
cols = ['Species']
data[cols] = data[cols].apply(lambda x: pd.factorize(x)[0] + 1)

# Splitting into Features and Labels
features = data.drop(columns=["Id", "Species"])
labels   = data['Species']

# Splitting into Train and Test Set
xTrain, xTest, yTrain, yTest = tts(features, labels, test_size=0.2, shuffle=True)

# Creating the model
model = DecisionTreeClassifier(max_depth=3, criterion='entropy')
model.fit(xTrain, yTrain)

# Predicting
tree_yPred = model.predict(xTest)

# Accuracy
print("Accuracy :", round(accuracy_score(yTest, tree_yPred)*100, 2), "%")

# Saving the model
joblib.dump(model, 'model.pkl')