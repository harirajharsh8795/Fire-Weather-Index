import pickle
from flask import Flask, request, render_template
import numpy as np

app = Flask(__name__)

# Load model and scaler
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predictdata', methods=['POST'])
def predict_data():
    try:
        Temperature = float(request.form.get('Temprature'))
        RH = float(request.form.get('RH'))
        WS = float(request.form.get('WS'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        data = [[Temperature, RH, WS, Rain, FFMC, DMC, ISI, Classes, Region]]
        scaled = standard_scaler.transform(data)
        result = ridge_model.predict(scaled)

        return render_template('home.html', results=round(result[0], 2))

    except Exception as e:
        return render_template('home.html', results=f"Error: {e}")

if __name__ == '__main__':
    app.run(debug=True)
