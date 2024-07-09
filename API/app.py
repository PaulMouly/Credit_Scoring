from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from xgboost import XGBClassifier

app = Flask(__name__)

# Chargement du modèle pré-entraîné
model = joblib.load('model/xgboost_model_reduced.pkl')  

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame(data)
    prediction = model.predict(df)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True, port=5001)


