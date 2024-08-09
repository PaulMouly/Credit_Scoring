def test_predict_with_valid_data(client):
    response = client.get('/predict?SK_ID_CURR=100002')
    assert response.status_code == 200
    assert 'SK_ID_CURR' in response.json
    assert 'prediction' in response.json
    assert response.json['prediction'] in [0, 1]  

def test_predict_with_invalid_data(client):
    # On teste ici pour voir comment l'API gère les entrées invalides
    response = client.get('/predict?SK_ID_CURR=invalid')
    assert response.status_code == 400
    assert "SK_ID_CURR invalid ne peut pas être converti en entier." in response.data
