import os
import sys
import pytest

# Ensure backend app is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Set dummy env vars so Config doesn't fail during tests
os.environ.setdefault('LLM_API_KEY', 'test-key')
os.environ.setdefault('ZEP_API_KEY', 'test-key')


@pytest.fixture
def app():
    """Create Flask test app."""
    from app import create_app
    application = create_app()
    application.config['TESTING'] = True
    return application


@pytest.fixture
def client(app):
    """Create Flask test client."""
    return app.test_client()


@pytest.fixture
def tmp_text_file(tmp_path):
    """Create a temporary text file for parsing tests."""
    p = tmp_path / "sample.txt"
    p.write_text("Hello World.\nThis is a test document.\nThird line here.", encoding="utf-8")
    return str(p)


@pytest.fixture
def tmp_md_file(tmp_path):
    """Create a temporary markdown file."""
    p = tmp_path / "sample.md"
    p.write_text("# Title\n\nParagraph one.\n\n## Section\n\nParagraph two.", encoding="utf-8")
    return str(p)
