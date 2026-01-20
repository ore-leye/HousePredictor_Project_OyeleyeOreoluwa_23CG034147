from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the saved model and scaler
model_path = os.path.join('model', 'house_price_model.pkl')
scaler_path = os.path.join('model', 'scaler.pkl')
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve data from form
    input_data = [float(x) for x in request.form.values()]
    
    # Scale input data
    final_features = scaler.transform([np.array(input_data)])
    
    # Predict
    prediction = model.predict(final_features)
    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text=f'Estimated House Price: ${output:,}')

if __name__ == "__main__":
    app.run(debug=True)