import pytest
from flask import Flask
from app import app as flask_app 

@pytest.fixture
def client():
    # Configure l'application pour les tests
    flask_app.config['TESTING'] = True
    with flask_app.test_client() as client:
        yield client
