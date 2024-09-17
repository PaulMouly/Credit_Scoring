# API Development

## Objectif du projet
Cette branche se concentre sur le développement et le déploiement de l'API pour l'outil de scoring crédit. L'API est conçue pour intégrer le modèle de prédiction de crédit dans une interface web, permettant ainsi des prédictions en temps réel via des requêtes. Le développement inclut également des tests unitaires pour assurer la robustesse et la fiabilité de l'API.

## Structure du projet

- `API/` : Contient les fichiers de l'API et les configurations nécessaires pour le déploiement.
  - `__pycache__/` : Répertoire généré automatiquement contenant les fichiers compilés Python.
  - `__init__.py` : Fichier d'initialisation pour le package Python.
  - `app.py` : Script principal de l'application Flask pour l'API.
  - `app.log` : Fichier de log pour l'application.
  - `conftest.py` : Configuration pour les tests avec pytest.
  - `test_api1.py` : Premier fichier de test pour l'API.
  - `test_api2.py` : Deuxième fichier de test pour l'API.
  - `tests/` : Répertoire contenant les tests unitaires.
- `data/` : Contient les données nécessaires pour les prédictions.
  - `X_predictionV1.csv` : Jeu de données utilisé pour les prédictions.
- `model/` : Contient les modèles pré-entraînés.
  - `xgboost_model.pkl` : Modèle XGBoost utilisé pour les prédictions.
- `static/` : Contient les fichiers statiques pour l'interface utilisateur.
  - `images/` : Dossier contenant les images utilisées dans l'interface utilisateur.
    - `prediction_time.png` : Image affichée dans l'interface utilisateur.
  - `styles.css` : Fichier CSS pour le style de l'interface utilisateur.
- `templates/` : Contient les fichiers de template HTML pour l'interface utilisateur.
  - `index.html` : Page d'accueil de l'application.
  - `predict.html` : Page de prédiction affichant les résultats.
- `Procfile` : Fichier de configuration pour le déploiement sur Heroku.
- `Dockerfile` : Fichier de configuration pour la création de l'image Docker de l'application.
- `heroku.yml` : Fichier de configuration pour le déploiement sur Heroku.
- `requirements.txt` : Liste des dépendances Python nécessaires pour le projet.

## Instructions

1. Clonez ce dépôt : `git clone https://github.com/votre-utilisateur/projet-api-development.git`
2. Installez les dépendances : `pip install -r requirements.txt`
3. Exécutez l'application Flask 
