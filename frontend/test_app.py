"""Basic tests for Flask application"""
import pytest
from app import app, allowed_file


@pytest.fixture
def client():
    """Create test client"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_app_exists():
    """Test Flask app initialization"""
    assert app is not None


def test_health_endpoint(client):
    """Test /health endpoint"""
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json['status'] == 'healthy'


def test_index_route(client):
    """Test index page loads"""
    response = client.get('/')
    assert response.status_code == 200


def test_allowed_file_valid():
    """Test valid file extensions"""
    assert allowed_file('image.png') == True
    assert allowed_file('photo.jpg') == True
    assert allowed_file('pic.jpeg') == True


def test_allowed_file_invalid():
    """Test invalid file extensions"""
    assert allowed_file('file.txt') == False
    assert allowed_file('script.exe') == False
    assert allowed_file('noextension') == False
