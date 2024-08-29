import os
import logging
import sys
import joblib
import pandas as pd
from flask import Flask, request, jsonify



app = Flask(__name__)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    handlers=[
                        logging.FileHandler("app.log", encoding="utf-8"),
                        logging.StreamHandler(sys.stdout)
                    ])

logger = logging.getLogger(__name__)

    
# Obtenir le chemin absolu du répertoire courant
current_dir = os.path.dirname(os.path.abspath(__file__))
# Spécifier le chemin relatif du fichier de données prétraitées
processed_data_path = os.path.join(current_dir, 'data', 'X_predictionV1.csv')
# Spécifier le chemin relatif du fichier modèle
model_path = os.path.join(current_dir, 'model', 'xgboost_model.pkl')
# Charger le modèle
try:
    model = joblib.load(model_path)
    logger.info("Modèle chargé avec succès")
except FileNotFoundError:
    logger.error(f"Le fichier modèle à l'emplacement {model_path} est introuvable.")
    model = None
except Exception as e:
    logger.error(f"Erreur lors du chargement du modèle : {e}")
    model = None

# Extraction des noms de colonnes utilisées pour l'entraînement
try:
    if model:
        cols_when_model_builds = model.get_booster().feature_names
    else:
        cols_when_model_builds = []
except Exception as e:
    logger.error(f"Erreur lors de l'extraction des noms de colonnes du modèle : {e}")
    cols_when_model_builds = []

# Définir le seuil de probabilité
threshold = 0.5  # Remplacez ceci par le seuil que vous avez déterminé

@app.route('/')
def home():
    return "Bienvenue dans l'application de prédiction"

@app.route('/predict', methods=['GET'])
def predict():
    try:
        sk_id_curr = request.args.get('SK_ID_CURR')

        if not sk_id_curr:
            app.logger.warning("SK_ID_CURR non fourni dans la requête")
            return jsonify({'error': 'Veuillez fournir SK_ID_CURR en paramètre.'}), 400

        app.logger.info("SK_ID_CURR reçu %s", {sk_id_curr})

        # Vérifier que SK_ID_CURR peut être converti en entier
        try:
            sk_id_curr = int(sk_id_curr)
            app.logger.info("Type de sk_id_curr = %s", {type(sk_id_curr)})
        except ValueError:
            app.logger.error("SK_ID_CURR %s ne peut pas être converti en entier.", sk_id_curr)
            return jsonify({'error': f'SK_ID_CURR {sk_id_curr} ne peut pas être converti en entier.'}), 400

        # Lire le fichier CSV en morceaux et filtrer les données
        data_found = False
        app.logger.info(os.path.exists(processed_data_path))
        try:
            for chunk in pd.read_csv(processed_data_path, chunksize=2000):
                if 'SK_ID_CURR' not in chunk.columns:
                    app.logger.error("'SK_ID_CURR' column is missing from the CSV.")
                    continue
                data_row = chunk[chunk['SK_ID_CURR'] == sk_id_curr]
                if not data_row.empty:
                    data_found = True
                    break
        except FileNotFoundError:
            app.logger.error(f"File not found: {processed_data_path}")
            return jsonify({'error': 'Fichier non trouvé.'}), 404
        except pd.errors.EmptyDataError:
            app.logger.error(f"No data: {processed_data_path} is empty or corrupted.")
            return jsonify({'error': 'Le fichier est vide ou corrompu.'}), 400
        except Exception as e:
            app.logger.error(f"An error occurred: {str(e)}")
            return jsonify({'error': 'Une erreur est survenue lors de la lecture du fichier.'}), 500
        
        if not data_found:
            app.logger.warning("Aucune donnée trouvée pour SK_ID_CURR %s", sk_id_curr)
            return jsonify({'error': f'Aucune donnée trouvée pour SK_ID_CURR {sk_id_curr}.'}), 404

        try:
            df = data_row.copy()
            # Vérifiez la forme des données avant la prédiction
            app.logger.info("Shape des données avant prédiction : %s", df.shape)

            # Réorganiser les colonnes selon l'ordre attendu par le modèle
            df = df[cols_when_model_builds]
        except KeyError as e:
            app.logger.error("Erreur lors de la réorganisation des colonnes: %s", e)
            return jsonify({'error': f"Les colonnes nécessaires pour le modèle sont manquantes : {str(e)}"}), 400
        except Exception as e:
            app.logger.error("Erreur lors de la préparation des données pour la prédiction: %s", e)
            return jsonify({'error': str(e)}), 400

        try:
            X_np = df.values  # Convertir en matrice NumPy

            # Faire la prédiction des probabilités
            predictions_proba = model.predict_proba(X_np)[:, 1]

            # Appliquer le seuil pour déterminer la classe
            prediction = (predictions_proba > threshold).astype(int)

            result = {
                'SK_ID_CURR': sk_id_curr,
                'prediction': int(prediction[0])
            }

            return jsonify(result), 200

        except Exception as e:
            app.logger.error(f"Erreur lors de la prédiction : {e}")
            return jsonify({'error': str(e)}), 400

    except Exception as e:
        app.logger.error(f"Erreur lors de la gestion de la requête : {e}")
        return jsonify({'error': str(e)}), 400



#def 

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)
