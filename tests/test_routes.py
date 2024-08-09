def test_home_route(client):
    response = client.get('/')
    assert response.status_code == 200
    assert "Bienvenue dans l'application de prédiction" in response.data

def test_predict_route_valid_data(client):
    response = client.get('/predict?SK_ID_CURR=100002')
    assert response.status_code == 200
    assert 'SK_ID_CURR' in response.json
    assert 'prediction' in response.json

def test_predict_route_missing_param(client):
    response = client.get('/predict')
    assert response.status_code == 400
    assert "Veuillez fournir SK_ID_CURR en paramètre." in response.data

def test_predict_route_invalid_param(client):
    response = client.get('/predict?SK_ID_CURR=invalid')
    assert response.status_code == 400
    assert "SK_ID_CURR invalid ne peut pas être converti en entier." in response.data

def test_predict_route_no_data(client):
    response = client.get('/predict?SK_ID_CURR=1')  
    assert response.status_code == 404
    assert "Aucune donnée trouvée pour SK_ID_CURR 1." in response.data
