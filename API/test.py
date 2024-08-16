import pandas as pd

# Chemin relatif vers le fichier CSV de test
CSV_FILE_PATH = 'data/X_predictionV1.csv'

# Test sur le data frame
def test_csv_loading():
    # Charger le fichier CSV
    df = pd.read_csv(CSV_FILE_PATH)

    # Vérifier que le DataFrame n'est pas vide
    assert df.shape[0] > 0, "Le DataFrame est vide"

    # Vérifier que la colonne : SK_ID_CURR existe bien  
    expected_columns = ['SK_ID_CURR'] 
    for col in expected_columns:
        assert col in df.columns, f"La colonne {col} est manquante dans le DataFrame" 



# Test de l'API
def test_predict_route_valid_data(client):
    response = client.get('/predict?SK_ID_CURR=100002')
    assert response.status_code == 200
    assert 'SK_ID_CURR' in response.json



