from flask import Flask, request, jsonify
import joblib
import pandas as pd
import sklearn

app = Flask(__name__)

# Chemin vers le modèle pré-entraîné
model_path = 'C:/Users/paulm/Documents/Projet 7/Projet7withCSV/model/xgboost_model.pkl'
# Chargement du modèle et du pipeline de prétraitement
model_path = 'model/xgboost_model.pkl'
preprocessing_path = 'C:/Users/paulm/Documents/Projet 7/Projet7withCSV/model/preprocessing_pipeline.pkl'

model = joblib.load(model_path)
preprocessing_pipeline = joblib.load(preprocessing_path)

# Définir la route pour les prédictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Récupérer les données JSON de la requête
        data = request.get_json()

        # Créer un DataFrame pandas à partir des données JSON
        df = pd.DataFrame(data, index=[0])

        # Appliquer le prétraitement
        df_encoded = df.drop(columns=['SK_ID_CURR'])  # Supprimer SK_ID_CURR
        for col in preprocessing_pipeline['drop_columns']:
            if col in df_encoded.columns:
                df_encoded.drop(columns=[col], inplace=True)  # Supprimer les colonnes à supprimer

        for col in preprocessing_pipeline['inf_to_nan']:
            if col in df_encoded.columns:
                df_encoded = df_encoded[~np.isinf(df_encoded[col])]  # Supprimer les lignes avec infinies

        df_encoded = pd.DataFrame(preprocessing_pipeline['encoder'].transform(df_encoded))  # Encoder
        df_encoded = pd.DataFrame(preprocessing_pipeline['imputer'].transform(df_encoded))  # Imputer

        # Ajouter SK_ID_CURR
        df_encoded['SK_ID_CURR'] = df['SK_ID_CURR'].values

        # Effectuer la prédiction avec le modèle
        X = df_encoded.drop(columns=['TARGET'])
        y_pred = model.predict(X)

        # Formater la réponse en JSON
        predictions = {'predictions': y_pred.tolist()}

        return jsonify(predictions)

    except Exception as e:
        return jsonify({'error': str(e)}), 400


