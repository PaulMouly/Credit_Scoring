import requests
import json

url = 'http://127.0.0.1:5001/predict'

def test_prediction_valid_id():
    data = {'SK_ID_CURR': 100001}
    json_data = json.dumps(data)
    response = requests.post(url, json=json_data)
    assert response.status_code == 200
    assert 'prediction' in response.json()

def test_prediction_invalid_id():
    data = {'SK_ID_CURR': 999999}
    json_data = json.dumps(data)
    response = requests.post(url, json=json_data)
    assert response.status_code == 404 or response.status_code == 400
    assert 'error' in response.json()

def test_prediction_missing_id():
    data = {}
    json_data = json.dumps(data)
    response = requests.post(url, json=json_data)
    assert response.status_code == 400
    assert 'error' in response.json()

def test_prediction_performance():
    data = {'SK_ID_CURR': 100001}
    json_data = json.dumps(data)
    response = requests.post(url, json=json_data)
    assert response.status_code == 200
    assert response.elapsed.total_seconds() < 2