import requests
import json

url = 'http://127.0.0.1:5001/predict'  # Assure-toi que l'URL correspond à celle où ton API est en cours d'exécution
data = {'SK_ID_CURR': 100001}  # Remplace par un SK_ID_CURR existant dans tes données

# Convertir les données en format JSON
json_data = json.dumps(data)

# Envoyer la requête POST à l'API
response = requests.post(url, json=json_data)

# Vérifier le statut de la réponse
if response.status_code == 200:
    print("Requête réussie ! Voici la réponse :")
    print(response.json())
else:
    print(f"Erreur lors de la requête : {response.status_code}")