import pytest
from flask import Flask
from app import app  

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_predict_route_valid_data(client):
    response = client.get('/predict?SK_ID_CURR=100002')
    assert response.status_code == 200
    assert 'SK_ID_CURR' in response.data.decode('utf-8')  # Vérifiez que la réponse contient SK_ID_CURR

def test_predict_route_missing_id(client):
    response = client.get('/predict')
    assert response.status_code == 200
    assert 'Veuillez fournir SK_ID_CURR en paramètre.' in response.data.decode('utf-8')

def test_predict_route_invalid_id(client):
    response = client.get('/predict?SK_ID_CURR=abc')
    assert response.status_code == 200
    assert 'SK_ID_CURR abc ne peut pas être converti en entier.' in response.data.decode('utf-8')

def test_predict_route_data_not_found(client):
    response = client.get('/predict?SK_ID_CURR=999999')
    assert response.status_code == 200
    assert 'Aucune donnée trouvée pour SK_ID_CURR 999999.' in response.data.decode('utf-8')

# Ajoutez d'autres tests si nécessaire pour couvrir différents cas d'erreur ou de validation