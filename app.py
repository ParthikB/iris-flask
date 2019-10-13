from flask import Flask, render_template, request
import numpy as np
from sklearn.externals import joblib

app = Flask(__name__)

# Loading the model
model_file = open('model.pkl', 'rb')
model = joblib.load(model_file)
labels = {1:"Iris-setosa", 2:"Iris-versicolor", 3:"Iris-virginica"}

# Defining the Home page
@app.route('/')
def home():
    return render_template('home.html')

# Defining the Prediction page
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        print("it's working")
        try:
            # Taking the inputs and saving them to a variable
            SepalLength = float(request.form['SepalLength'])
            SepalWidth  = float(request.form['SepalWidth'])
            PetalLength = float(request.form['PetalLength'])
            PetalWidth  = float(request.form['PetalWidth'])
            
            # Converting the inputs into a numpy array
            pred_args = np.array([SepalLength, SepalWidth, PetalLength, PetalWidth]).reshape(1, -1)
            
            # Predicting the Label
            model_prediction = model.predict(pred_args)[0]
            model_prediction = labels[model_prediction]

        except:
            return 'Invalid Values entered!'

    return render_template('predict.html', prediction = model_prediction)


if __name__ == '__main__':
    app.run()