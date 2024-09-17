import pytest
from flask import Flask
from app import app

@pytest.fixture
def client():
    app.testing = True
    return app.test_client()

def test_home(client):
    response = client.get('/')
    assert response.status_code == 200
    assert 'Bienvenue dans l\'Application de Prédiction' in response.data.decode('utf-8')

def test_predict_valid(client):
    response = client.get('/predict?SK_ID_CURR=1')
    assert response.status_code == 200
    assert 'Résultat de la Prédiction' in response.data.decode('utf-8')

def test_predict_valid_prediction(client):
    sk_id_curr = 100002  
    expected_prediction = 1  

    response = client.get(f'/predict?SK_ID_CURR={sk_id_curr}')
    assert response.status_code == 200

    assert f'Prédiction : {expected_prediction}' in response.data.decode('utf-8')
