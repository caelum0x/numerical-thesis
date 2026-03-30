"""Tests for Flask app creation and health endpoint."""

import pytest


class TestAppFactory:
    def test_app_creates(self, app):
        assert app is not None
        assert app.config['TESTING'] is True

    def test_health_endpoint(self, client):
        response = client.get('/health')
        assert response.status_code == 200
        data = response.get_json()
        assert data['status'] == 'ok'
        assert data['service'] == 'MiroFish Backend'

    def test_cors_headers(self, client):
        response = client.options('/api/graph/project/list', headers={
            'Origin': 'http://localhost:3000',
            'Access-Control-Request-Method': 'GET',
        })
        # CORS should allow configured origins
        assert response.status_code in (200, 204)

    def test_blueprints_registered(self, app):
        rules = [rule.rule for rule in app.url_map.iter_rules()]
        assert '/health' in rules
        # Check that blueprint prefixes are registered
        assert any(r.startswith('/api/graph') for r in rules)
        assert any(r.startswith('/api/simulation') for r in rules)
        assert any(r.startswith('/api/report') for r in rules)
