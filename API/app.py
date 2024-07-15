
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier
import joblib
import os
import logging

app = Flask(__name__)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    handlers=[
                        logging.FileHandler("app.log", encoding="utf-8"),
                        logging.StreamHandler()
                    ])

logger = logging.getLogger(__name__)

# Chargement du modèle XGBoost
model_path = 'C:/Users/paulm/Documents/Projet 7/Projet7withCSV/model/xgboost_model.pkl'
model = joblib.load(model_path)

# Extraction des noms de colonnes utilisées pour l'entraînement
cols_when_model_builds = model.get_booster().feature_names

# Chargement des données originales pour la correspondance SK_ID_CURR
chemin_dossier = "C:/Users/paulm/Documents/Projet 7/Projet7withCSV/data/"
df_original = pd.read_csv(os.path.join(chemin_dossier, 'processed_data.csv'))

@app.route('/predict', methods=['GET'])
def predict():
    try:
        sk_id_curr = request.args.get('SK_ID_CURR')

        if not sk_id_curr:
            logger.warning("SK_ID_CURR non fourni dans la requête")
            return jsonify({'error': 'Veuillez fournir SK_ID_CURR en paramètre.'}), 400

        logger.info(f"SK_ID_CURR reçu : {sk_id_curr}")

        data_row = df_original[df_original['SK_ID_CURR'] == int(sk_id_curr)]

        if data_row.empty:
            logger.warning(f"Aucune donnée trouvée pour SK_ID_CURR {sk_id_curr}")
            return jsonify({'error': f'Aucune donnée trouvée pour SK_ID_CURR {sk_id_curr}.'}), 404

        df = data_row.copy()
        df.drop(columns=['ORGANIZATION_TYPE', 'OCCUPATION_TYPE'], inplace=True)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        ordinal_encoder = OrdinalEncoder()
        categorical_cols = df.select_dtypes(include='object').columns
        df[categorical_cols] = ordinal_encoder.fit_transform(df[categorical_cols])

        df = df[cols_when_model_builds]  # Réorganiser les colonnes selon l'ordre attendu

        # Vérifiez la forme des données avant la prédiction
        logger.info(f"Shape des données avant prédiction : {df.shape}")

        # Vérifiez et gérez les valeurs manquantes si nécessaire
        if df.isnull().values.any():
            logger.warning("Il y a des valeurs manquantes dans les données à prédire.")
            df.fillna(0, inplace=True)  # Par exemple, remplissez les valeurs manquantes avec zéro

        X = df.drop(columns=['SK_ID_CURR'])
        X_np = X.values  # Convertir en matrice NumPy

        prediction = model.predict(X_np)

        result = {
            'SK_ID_CURR': sk_id_curr,
            'prediction': int(prediction[0])
        }

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Erreur lors de la prédiction : {e}")
        return jsonify({'error': str(e)}), 400
