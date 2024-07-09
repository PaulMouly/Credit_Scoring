import requests
import json

url = 'http://127.0.0.1:5001/predict'
data = {'SK_ID_CURR': 100001}  # Remplace par un SK_ID_CURR existant dans tes données

# Convertir les données en format JSON
json_data = json.dumps(data)

# Envoyer la requête POST à l'API
response = requests.post(url, json=json_data)

# Afficher la réponse de l'API
print(response.json())
