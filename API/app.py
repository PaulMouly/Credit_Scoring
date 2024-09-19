import os
import logging
import sys
import joblib
import pandas as pd
from flask import Flask, request, render_template, jsonify

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
threshold = 0.5  

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        sk_id_curr = request.form.get('SK_ID_CURR')
    else:
        sk_id_curr = request.args.get('SK_ID_CURR')

    if not sk_id_curr:
        logger.warning("SK_ID_CURR non fourni dans la requête")
        return render_template('predict.html', error='Veuillez fournir SK_ID_CURR en paramètre.')

    try:
        sk_id_curr = int(sk_id_curr)
    except ValueError:
        logger.error(f"SK_ID_CURR {sk_id_curr} ne peut pas être converti en entier.")
        return render_template('predict.html', error=f'SK_ID_CURR {sk_id_curr} ne peut pas être converti en entier.')

    logger.info(f"SK_ID_CURR reçu : {sk_id_curr}")

    data_found = False
    try:
        for chunk in pd.read_csv(processed_data_path, chunksize=2000):
            if 'SK_ID_CURR' not in chunk.columns:
                logger.error("'SK_ID_CURR' column is missing from the CSV.")
                continue
            data_row = chunk[chunk['SK_ID_CURR'] == sk_id_curr]
            if not data_row.empty:
                data_found = True
                break
    except FileNotFoundError:
        logger.error(f"Fichier non trouvé : {processed_data_path}")
        return render_template('predict.html', error='Fichier non trouvé.')
    except pd.errors.EmptyDataError:
        logger.error(f"Le fichier est vide ou corrompu : {processed_data_path}")
        return render_template('predict.html', error='Le fichier est vide ou corrompu.')
    except Exception as e:
        logger.error(f"Erreur lors de la lecture du fichier : {str(e)}")
        return render_template('predict.html', error='Une erreur est survenue lors de la lecture du fichier.')

    if not data_found:
        logger.warning(f"Aucune donnée trouvée pour SK_ID_CURR {sk_id_curr}")
        return render_template('predict.html', error=f'Aucune donnée trouvée pour SK_ID_CURR {sk_id_curr}.')

    try:
        df = data_row.copy()
        df = df[cols_when_model_builds]
        X_np = df.values
        predictions_proba = model.predict_proba(X_np)[:, 1]
        prediction = (predictions_proba > threshold).astype(int)
        ###result = int(prediction[0])
        result_text = "crédit validé" if int(prediction[0]) == 0 else "crédit non validé"
        ###return render_template('predict.html', sk_id_curr=sk_id_curr, prediction=result)
        return render_template('predict.html', sk_id_curr=sk_id_curr, prediction=result_text)

    except Exception as e:
        logger.error(f"Erreur lors de la prédiction : {e}")
        return render_template('predict.html', error=str(e))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)
