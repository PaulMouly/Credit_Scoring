# Projet 7 - Scoring Crédit avec XGBoost

## Objectif du Projet
Le but de ce projet est de développer un outil de scoring crédit pour l'entreprise **Prêt à dépenser**, capable de prédire la probabilité qu’un client rembourse son crédit. Cet outil classifie chaque demande de crédit en deux catégories : **crédit accordé** ou **crédit refusé**, en fonction des caractéristiques des clients.

Dans cette branche dédiée à **XGBoost**, l'objectif est de créer, optimiser et déployer un modèle de classification robuste à partir de diverses sources de données.

## Structure de la Branche XGBoost
- **`data/`** : Contient les jeux de données utilisés pour l'entraînement et la prédiction :
  - `X_predictionV0.csv`, `X_predictionV1.csv`, `X_predictionV2.csv`, etc.
  - `processed_data.csv`, `processed_data_test.csv` pour les données de base prétraitées.
- **`scripts/`** : Contient les scripts Python utilisés pour le prétraitement, l'entraînement et l'évaluation des modèles :
  - `preprocessing/` : pipeline de prétraitement des données.
  - `preprocessing_pipeline.pkl` : pipeline enregistré.
  - `notebook_xg.ipynb` : exploration des résultats XGBoost.
  - **`model/`** : Contient les modèles XGBoost entraînés :
  - `xgboost_model.pkl` : modèle de base.
  - `xgboost_model_optimized.pkl` : modèle optimisé avec GridSearchCV.
  - `xgboost_model_with_smote_tomek_gridsearch.pkl` : modèle avec rééchantillonnage SMOTE...
- **`requirements.txt`** : Liste des dépendances Python nécessaires au projet.

## Modèles Entraînés
Cette branche contient plusieurs versions du modèle XGBoost entraînées et optimisées :
- **Modèle de base :** `xgboost_model.pkl`
- **Modèle optimisé avec GridSearchCV :** `xgboost_model_optimized.pkl`
- **Modèle réduit avec sélection de variables :** `xgboost_model_reduced.pkl`
- **Modèle avec sur-échantillonnage SMOTE et Tomek :** `xgboost_model_with_smote_tomek_gridsearch.pkl`

## Instructions
1. Clonez ce dépôt : `git clone https://github.com/votre-utilisateur/projet-7.git`
2. Installez les dépendances : `pip install -r requirements.txt`
3. Exécutez les scripts dans `scripts/` pour entraîner les modèles.
