from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os
import numpy as np
import xgboost as xgb

app = Flask(__name__)

# Chemin vers le modèle pré-entraîné
model_path = 'C:/Users/paulm/Documents/Projet 7/xgboost_model.pkl'

# Charger le modèle pré-entraîné
model = None
if os.path.exists(model_path):
    try:
        model = joblib.load(model_path)
        print(f"Modèle chargé avec succès depuis {model_path}.")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle : {str(e)}")
else:
    print(f"Le modèle {model_path} n'a pas été trouvé.")
    
if model is None:
    print("Impossible de continuer sans charger le modèle. Assurez-vous que le chemin du modèle est correct et réessayez.")
    exit(1)

@app.route('/predict', methods=['POST'])
@app.route('/predict', methods=['POST'])
def predict():
    data_request = request.json

    # Liste des features attendues pour la prédiction
    features = [
        'FLAG_EMP_PHONE',
        'CC_CNT_DRAWINGS_CURRENT_MEAN',
        'CC_AMT_RECEIVABLE_PRINCIPAL_MEAN',
        # 'NAME_EDUCATION_TYPE',  # Exclu temporairement
        'CC_MONTHS_BALANCE_SIZE',
        'CC_AMT_BALANCE_MEAN',
        'PREV_NAME_YIELD_GROUP_low_action_MEAN',
        'PREV_DAYS_FIRST_DRAWING_MIN'
    ]

    # Vérifier si les features attendues sont dans la requête
    for feature in features:
        if feature not in data_request:
            return jsonify({'error': f'{feature} est requis dans la requête'}), 400

    # Préparer les données pour la prédiction
    data_for_prediction = {
        'FLAG_EMP_PHONE': float(data_request['FLAG_EMP_PHONE']),
        'CC_CNT_DRAWINGS_CURRENT_MEAN': float(data_request['CC_CNT_DRAWINGS_CURRENT_MEAN']),
        'CC_AMT_RECEIVABLE_PRINCIPAL_MEAN': float(data_request['CC_AMT_RECEIVABLE_PRINCIPAL_MEAN']),
        # 'NAME_EDUCATION_TYPE': float(data_request['NAME_EDUCATION_TYPE']),  # Exclu temporairement
        'CC_MONTHS_BALANCE_SIZE': float(data_request['CC_MONTHS_BALANCE_SIZE']),
        'CC_AMT_BALANCE_MEAN': float(data_request['CC_AMT_BALANCE_MEAN']),
        'PREV_NAME_YIELD_GROUP_low_action_MEAN': float(data_request['PREV_NAME_YIELD_GROUP_low_action_MEAN']),
        'PREV_DAYS_FIRST_DRAWING_MIN': float(data_request['PREV_DAYS_FIRST_DRAWING_MIN'])
    }

    # Créer un DataFrame à partir des données pour la prédiction
    df = pd.DataFrame([data_for_prediction])

    # Faire la prédiction avec le modèle chargé
    try:
        prediction_proba = model.predict_proba(df)[:, 1]
        prediction_class = (prediction_proba > 0.5).astype(int)  # Utilisation du seuil métier (0.5 ici)

        return jsonify({
            'prediction': {
                'proba_default': float(prediction_proba[0]),
                'class': 'accepté' if prediction_class[0] == 0 else 'refusé'  # 0 pour accepté, 1 pour refusé
            }
        })

    except Exception as e:
        print(f"Erreur lors de la prédiction : {str(e)}")
        return jsonify({'error': 'Erreur lors de la prédiction'}), 500



