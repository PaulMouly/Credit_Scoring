
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier
import joblib
import os


app = Flask(__name__)

# Chargement du modèle XGBoost
model_path = 'C:/Users/paulm/Documents/Projet 7/Projet7withCSV/model/xgboost_model.pkl'
model = joblib.load(model_path)

# Chargement des données originales pour la correspondance SK_ID_CURR
chemin_dossier = "C:/Users/paulm/Documents/Projet 7/Projet7withCSV/data/"
df_original = pd.read_csv(os.path.join(chemin_dossier, 'processed_data.csv'))

# Définir la route pour les prédictions basées sur SK_ID_CURR
@app.route('/predict', methods=['GET'])
def predict():
    try:
        # Récupérer SK_ID_CURR à partir des paramètres de la requête
        sk_id_curr = request.args.get('SK_ID_CURR')

        # Vérifier si SK_ID_CURR est présent
        if not sk_id_curr:
            return jsonify({'error': 'Veuillez fournir SK_ID_CURR en paramètre.'}), 400

        print(f"SK_ID_CURR reçu : {sk_id_curr}")

        # Récupérer les données correspondant à SK_ID_CURR depuis df_original
        data_row = df_original[df_original['SK_ID_CURR'] == int(sk_id_curr)]

        if data_row.empty:
            return jsonify({'error': f'Aucune donnée trouvée pour SK_ID_CURR {sk_id_curr}.'}), 404

        print("Données trouvées :")
        print(data_row)

        # Prétraitement des données pour la prédiction
        df = data_row.copy()  # Copie des données pour éviter les modifications sur les données originales

        # Suppression des colonnes avec trop de valeurs uniques
        colonnes_a_supprimer = ['ORGANIZATION_TYPE', 'OCCUPATION_TYPE']
        df.drop(columns=colonnes_a_supprimer, inplace=True)

        print("Données après suppression des colonnes :")
        print(df)

        # Gestion des valeurs infinies
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Encodage des variables catégorielles avec OrdinalEncoder
        ordinal_encoder = OrdinalEncoder()
        categorical_cols = df.select_dtypes(include='object').columns
        df[categorical_cols] = ordinal_encoder.fit_transform(df[categorical_cols])

        print("Données encodées :")
        print(df)

        # Gestion des valeurs manquantes avec SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

        print("Données après imputation :")
        print(df)

        # Faire la prédiction avec le modèle XGBoost
        X = df.drop(columns=['SK_ID_CURR'])
        prediction = model.predict(X)

        # Formater la réponse en JSON
        result = {
            'SK_ID_CURR': sk_id_curr,
            'prediction': int(prediction[0])
        }

        return jsonify(result), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 400


