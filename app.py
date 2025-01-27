from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import pickle
import json

# Load the pre-trained model (ensure you have saved your model as 'model.pkl')
with open('D:/Dimond_prise_prediction_project/model/model_1.pkl', 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    carat = float(request.form['carat'])
    cut = int(request.form['cut'])
    color = int(request.form['color'])
    clarity = int(request.form['clarity'])
    depth = float(request.form['depth'])
    table = float(request.form['table'])
    x = float(request.form['x'])
    y = float(request.form['y'])
    z = float(request.form['z'])
    features  = [carat, cut, color, clarity, depth, table, x, y, z]
    final_features = [np.array(features)]
    # prediction using the loaded model file
    prediction = model.predict(final_features)
    
    # return jsonify({'predicted_price is -> ': float(prediction[0])})
    return render_template('result.html', predict='{}'.format(float(prediction[0])))


if __name__ == '__main__':
    app.run(debug=True)
