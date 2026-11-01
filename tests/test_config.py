"""Configuration settings tests.

Proves that embeddings settings read the environment as documented and handle
edge cases correctly.
"""

from __future__ import annotations

import os
import re
from dataclasses import FrozenInstanceError

import pytest

from embeddings.config import DEFAULT_DIMENSION, DEFAULT_MODEL, SERVICE_VERSION, get_settings


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    """No EMBEDDINGS_* variable from the host leaks into a test."""
    for name in [k for k in os.environ if k.startswith("EMBEDDINGS_")]:
        monkeypatch.delenv(name, raising=False)


class TestConfigDefaults:
    """Test default values when no environment variables are set."""

    def test_defaults_with_no_env_vars(self, monkeypatch):
        """With none set: use documented defaults."""

        settings = get_settings()

        assert settings.host == "0.0.0.0"
        assert settings.port == 8030
        assert settings.ollama_url == "http://ollama:11434"
        assert settings.model == DEFAULT_MODEL
        assert settings.dimension == DEFAULT_DIMENSION
        assert settings.admin_key == ""


class TestConfigEnvironmentVariables:
    """Test that all environment variables are read correctly."""

    def test_all_variables_set(self, monkeypatch):
        """Every variable set: each field takes it; port and dimension are ints."""

        # Set all variables
        monkeypatch.setenv("EMBEDDINGS_HOST", "1.2.3.4")
        monkeypatch.setenv("EMBEDDINGS_PORT", "8031")
        monkeypatch.setenv("EMBEDDINGS_OLLAMA_URL", "http://example:9999")
        monkeypatch.setenv("EMBEDDINGS_MODEL", "test-model:latest")
        monkeypatch.setenv("EMBEDDINGS_DIMENSION", "768")
        monkeypatch.setenv("EMBEDDINGS_ADMIN_KEY", "test-key-12345678")

        settings = get_settings()

        assert settings.host == "1.2.3.4"
        assert settings.port == 8031
        assert settings.ollama_url == "http://example:9999"
        assert settings.model == "test-model:latest"
        assert settings.dimension == 768
        assert settings.admin_key == "test-key-12345678"

    def test_values_are_stripped(self, monkeypatch):
        """Values are stripped: whitespace is removed."""

        # Test with whitespace
        monkeypatch.setenv("EMBEDDINGS_HOST", "  1.2.3.4  ")
        monkeypatch.setenv("EMBEDDINGS_PORT", "  8031  ")
        monkeypatch.setenv("EMBEDDINGS_OLLAMA_URL", "  http://example:9999  ")
        monkeypatch.setenv("EMBEDDINGS_MODEL", "  test-model:latest  ")
        monkeypatch.setenv("EMBEDDINGS_DIMENSION", "  768  ")
        monkeypatch.setenv("EMBEDDINGS_ADMIN_KEY", "  test-key-12345678  ")

        settings = get_settings()

        assert settings.host == "1.2.3.4"
        assert settings.port == 8031
        assert settings.ollama_url == "http://example:9999"
        assert settings.model == "test-model:latest"
        assert settings.dimension == 768
        assert settings.admin_key == "test-key-12345678"

    def test_ollama_url_trailing_slashes_removed(self, monkeypatch):
        """URL trailing slashes are stripped correctly."""

        # Test single trailing slash
        monkeypatch.setenv("EMBEDDINGS_OLLAMA_URL", "http://x:1/")
        settings = get_settings()
        assert settings.ollama_url == "http://x:1"

        # Test multiple trailing slashes
        monkeypatch.setenv("EMBEDDINGS_OLLAMA_URL", "http://x:1///")
        settings = get_settings()
        assert settings.ollama_url == "http://x:1"

    def test_invalid_port_raises_value_error(self, monkeypatch):
        """EMBEDDINGS_PORT="abc" raises ValueError."""

        monkeypatch.setenv("EMBEDDINGS_PORT", "abc")

        with pytest.raises(ValueError, match="invalid literal for int"):
            get_settings()

    def test_invalid_dimension_raises_value_error(self, monkeypatch):
        """EMBEDDINGS_DIMENSION="1.5" raises ValueError."""

        monkeypatch.setenv("EMBEDDINGS_DIMENSION", "1.5")

        with pytest.raises(ValueError, match="invalid literal for int"):
            get_settings()


class TestConfigImmutability:
    """Test that Settings is frozen and immutable."""

    def test_settings_is_frozen(self, monkeypatch):
        """Settings is frozen: assigning a field raises FrozenInstanceError."""

        settings = get_settings()

        with pytest.raises(FrozenInstanceError):
            settings.host = "new-host"

        with pytest.raises(FrozenInstanceError):
            settings.port = 9999

        with pytest.raises(FrozenInstanceError):
            settings.ollama_url = "http://new"

        with pytest.raises(FrozenInstanceError):
            settings.model = "new-model"

        with pytest.raises(FrozenInstanceError):
            settings.dimension = 512

        with pytest.raises(FrozenInstanceError):
            settings.admin_key = "new-key"


class TestServiceVersion:
    """Test that SERVICE_VERSION matches CHANGELOG.md."""

    def test_service_version_matches_changelog(self):
        """SERVICE_VERSION matches the first ## version heading in CHANGELOG.md."""
        changelog_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "CHANGELOG.md")

        try:
            with open(changelog_path, encoding="utf-8") as f:
                changelog_content = f.read()
        except FileNotFoundError:
            pytest.skip("CHANGELOG.md not found")

        # Look for version headings in the form "## 0.1.4" or "## [0.1.4]"
        version_pattern = r"^##\s*(?:\[)?(\d+\.\d+\.\d+)(?:\])?"
        matches = re.findall(version_pattern, changelog_content, re.MULTILINE)

        if not matches:
            pytest.skip("No version heading found in CHANGELOG.md")

        first_version = matches[0]
        assert first_version == SERVICE_VERSION, (
            f"SERVICE_VERSION ({SERVICE_VERSION}) does not match "
            f"first version in CHANGELOG.md ({first_version})"
        )
