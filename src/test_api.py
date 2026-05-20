"""
Unit tests for the IDS API
"""
import pytest
import sys
import os

# Add src to path so we can import api
sys.path.insert(0, os.path.dirname(__file__))

from api import app, PRODUCTION_MODE

@pytest.fixture
def client():
    """Create a test client for the API"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_home_endpoint(client):
    """Test the home endpoint returns 200"""
    response = client.get('/')
    assert response.status_code == 200
    assert b'IDS API is running' in response.data

def test_predict_endpoint_no_data(client):
    """Test predict endpoint with no data returns 400"""
    response = client.post('/predict', json={})
    assert response.status_code in [400, 503]  # 400 for missing data, 503 if test mode

def test_predict_endpoint_missing_features(client):
    """Test predict endpoint with missing features key"""
    response = client.post('/predict', json={"wrong_key": []})
    assert response.status_code in [400, 503]

def test_predict_endpoint_test_mode(client):
    """Test predict returns error in test mode (no model files)"""
    if not PRODUCTION_MODE:
        response = client.post('/predict', json={"features": [0.1] * 39})
        assert response.status_code == 503
        data = response.get_json()
        assert 'test mode' in data['error'].lower()

def test_production_mode_prediction(client):
    """Test prediction works in production mode"""
    if PRODUCTION_MODE:
        # Valid request with 39 features
        features = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                   0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                   0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                   0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        response = client.post('/predict', json={"features": features})
        assert response.status_code == 200
        data = response.get_json()
        assert 'prediction' in data
        assert 'result' in data
        assert 'confidence' in data
        assert data['result'] in ['ATTACK', 'NORMAL']
