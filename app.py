from flask import Flask, request, jsonify
import pandas as pd
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

# Obtenir le chemin absolu du répertoire courant
current_dir = os.path.dirname(os.path.abspath(__file__))

# Spécifier le chemin relatif du fichier modèle
model_path = os.path.join(current_dir, 'model', 'xgboost_model.pkl')
# Charger le modèle
try:
    model = joblib.load(model_path)
    logger.info("Modèle chargé avec succès")
except FileNotFoundError:
    logger.error(f"Le fichier modèle à l'emplacement {model_path} est introuvable.")
    model = None

# Spécifier le chemin relatif du fichier de données prétraitées
processed_data_path = os.path.join(current_dir, 'data', 'X_prediction.csv')
# Charger les données prétraitées
try:
    df_prediction = pd.read_csv(processed_data_path)
    logger.info("Données prétraitées chargées avec succès")
except FileNotFoundError:
    logger.error(f"Le fichier de données prétraitées à l'emplacement {processed_data_path} est introuvable.")
    df_prediction = None

# Extraction des noms de colonnes utilisées pour l'entraînement
if model:
    cols_when_model_builds = model.get_booster().feature_names
else:
    cols_when_model_builds = []

@app.route('/predict', methods=['GET'])
def predict():
    try:
        sk_id_curr = request.args.get('SK_ID_CURR')

        if not sk_id_curr:
            logger.warning("SK_ID_CURR non fourni dans la requête")
            return jsonify({'error': 'Veuillez fournir SK_ID_CURR en paramètre.'}), 400

        logger.info(f"SK_ID_CURR reçu : {sk_id_curr}")

        # Récupérer les données correspondant à SK_ID_CURR depuis df_prediction
        data_row = df_prediction[df_prediction['SK_ID_CURR'] == int(sk_id_curr)]

        if data_row.empty:
            logger.warning(f"Aucune donnée trouvée pour SK_ID_CURR {sk_id_curr}")
            return jsonify({'error': f'Aucune donnée trouvée pour SK_ID_CURR {sk_id_curr}.'}), 404

        df = data_row.copy()

        # Vérifiez la forme des données avant la prédiction
        logger.info(f"Shape des données avant prédiction : {df.shape}")

        # Réorganiser les colonnes selon l'ordre attendu par le modèle
        df = df[cols_when_model_builds]

        X_np = df.values  # Convertir en matrice NumPy

        prediction = model.predict(X_np)

        result = {
            'SK_ID_CURR': sk_id_curr,
            'prediction': int(prediction[0])
        }

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Erreur lors de la prédiction : {e}")
        return jsonify({'error': str(e)}), 400


