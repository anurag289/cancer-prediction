from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)
model = pickle.load(open('Knearforcancer.pkl', 'rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


standard_to = StandardScaler()
@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        radius=float(request.form['radius'])
        perimeter=float(request.form['perimeter'])
        prediction=model.predict([[radius,perimeter]])
        output=prediction[0]
        if output=='M':
            return render_template('index.html',prediction_text="This is Malignant Cancer")  
        else:
            return render_template('index.html',prediction_text="This is Benign Cancer")
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)

