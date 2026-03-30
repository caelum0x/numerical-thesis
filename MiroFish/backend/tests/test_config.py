"""Tests for Config validation and defaults."""

import os
import pytest
from app.config import Config


class TestConfig:
    def test_validate_passes_with_keys(self):
        os.environ['LLM_API_KEY'] = 'test-key'
        os.environ['ZEP_API_KEY'] = 'test-key'
        # Re-read config values
        Config.LLM_API_KEY = os.environ.get('LLM_API_KEY')
        Config.ZEP_API_KEY = os.environ.get('ZEP_API_KEY')
        errors = Config.validate()
        assert errors == []

    def test_validate_fails_without_llm_key(self):
        original = Config.LLM_API_KEY
        Config.LLM_API_KEY = None
        errors = Config.validate()
        assert any("LLM_API_KEY" in e for e in errors)
        Config.LLM_API_KEY = original

    def test_validate_fails_without_zep_key(self):
        original = Config.ZEP_API_KEY
        Config.ZEP_API_KEY = None
        errors = Config.validate()
        assert any("ZEP_API_KEY" in e for e in errors)
        Config.ZEP_API_KEY = original

    def test_default_values(self):
        assert Config.MAX_CONTENT_LENGTH == 50 * 1024 * 1024
        assert Config.DEFAULT_CHUNK_SIZE == 500
        assert Config.DEFAULT_CHUNK_OVERLAP == 50
        assert 'pdf' in Config.ALLOWED_EXTENSIONS
        assert 'md' in Config.ALLOWED_EXTENSIONS
        assert 'txt' in Config.ALLOWED_EXTENSIONS

    def test_oasis_actions_defined(self):
        assert len(Config.OASIS_TWITTER_ACTIONS) > 0
        assert 'CREATE_POST' in Config.OASIS_TWITTER_ACTIONS
        assert len(Config.OASIS_REDDIT_ACTIONS) > 0
        assert 'CREATE_POST' in Config.OASIS_REDDIT_ACTIONS
