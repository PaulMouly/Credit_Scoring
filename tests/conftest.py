import sys
import os
import pytest

# Ajouter le répertoire 'API' au sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'API')))

from API import app

@pytest.fixture
def client():
    app.app.config['TESTING'] = True
    with app.app.test_client() as client:
        yield client


