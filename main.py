from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load your trained ML model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve input values from form
        features = [
            float(request.form['age']),
            int(request.form['anaemia']),
            float(request.form['creatinine_phosphokinase']),
            int(request.form['diabetes']),
            float(request.form['ejection_fraction']),
            int(request.form['high_blood_pressure']),
            float(request.form['platelets']),
            float(request.form['serum_creatinine']),
            float(request.form['serum_sodium']),
            int(request.form['sex']),
            int(request.form['smoking']),
            int(request.form['time'])
        ]

        # Reshape input and predict
        prediction = model.predict([features])[0]
        result = "🟥 High Risk of Death" if prediction == 1 else "🟩 Likely Survival"

        return render_template("index.html", result=result)
    except Exception as e:
        # Handle errors gracefully
        return render_template("index.html", result=f"⚠️ Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)