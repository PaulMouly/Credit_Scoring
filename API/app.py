from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os
import xgboost as xgb


app = Flask(__name__)

# Chemin vers le modèle pré-entraîné réduit
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model', 'xgboost_model_reduced.pkl'))

# Vérifier si le modèle existe
if not os.path.exists(model_path):
    print(f"Le modèle {model_path} n'a pas été trouvé.")
    exit(1)

# Charger le modèle pré-entraîné
try:
    model = joblib.load(model_path)
except Exception as e:
    print(f"Erreur lors du chargement du modèle : {str(e)}")
    exit(1)

# Chemin vers le fichier CSV de données traitées
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed_data.csv'))

# Charger les données à partir du fichier CSV
try:
    data = pd.read_csv(data_path)
except FileNotFoundError:
    print(f"Le fichier {data_path} n'a pas été trouvé.")
    data = None
except Exception as e:
    print(f"Erreur lors du chargement des données : {str(e)}")
    data = None

@app.route('/predict', methods=['POST'])
def predict():
    if data is None:
        return jsonify({'error': 'Fichier de données non trouvé'})

    data_request = request.json
    sk_id_curr = data_request.get('SK_ID_CURR')

    # Vérifier si SK_ID_CURR existe dans les données chargées
    if sk_id_curr in data['SK_ID_CURR'].values:
        # Récupérer les données associées à SK_ID_CURR
        data_for_prediction = data[data['SK_ID_CURR'] == sk_id_curr].drop(columns=['SK_ID_CURR'])
        df = pd.DataFrame(data_for_prediction)
        
        # Faire la prédiction avec ton modèle réduit
        try:
            prediction = model.predict(df)
            return jsonify({'prediction': prediction.tolist()})
        except Exception as e:
            print(f"Erreur lors de la prédiction : {str(e)}")
            return jsonify({'error': 'Erreur lors de la prédiction'})

    else:
        return jsonify({'error': 'SK_ID_CURR non trouvé dans les données'})

if __name__ == '__main__':
    app.run(debug=True, port=5002)

