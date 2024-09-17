# Projet 7 - Branche Random Forest

## Objectif de la branche

Cette branche a été utilisée pour expérimenter avec un modèle Random Forest dans le cadre du projet de scoring crédit. Bien que le modèle Random Forest ait été testé, il s'est avéré moins performant et moins pertinent par rapport à d'autres approches, notamment XGBoost. Cette branche ne sera donc pas conservée dans la version finale du projet.

## Structure de la branche

- `data/` : Contient les données prétraitées utilisées pour l'entraînement du modèle, ancienne version.
- `scripts/` : Contient le script pour l'entraînement et l'évaluation du modèle Random Forest.
- `model/` : Contient le modèle Random Forest sauvegardé après entraînement.
  - `random_forest_model.pkl` : Fichier du modèle Random Forest.
- `random_forest.ipynb` : Notebook utilisé pour l'entraînement et l'analyse du modèle Random Forest.
- `.gitattributes` : Fichier de configuration Git pour gérer les attributs spécifiques des fichiers.
- `.gitignore` : Fichier listant les fichiers et dossiers à ignorer dans le dépôt Git.
- `README.md` : Ce fichier de documentation.
- `requirements.txt` : Fichier listant les dépendances nécessaires pour exécuter le modèle.

## Instructions

1. Clonez ce dépôt : `git clone https://github.com/votre-utilisateur/projet-7-rf.git`
2. Installez les dépendances : `pip install -r requirements.txt`
3. Exécutez le notebook `random_forest.ipynb` pour analyser et entraîner le modèle.

## Remarque

Cette branche a servi à tester le modèle Random Forest. Le modèle n'étant pas retenu dans la version finale du projet.

