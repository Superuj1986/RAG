
from reading.config import AppConfig


def test_google_api_key_alias():
    config = AppConfig(
        _env_file=None,
        GOOGLE_API_KEY="test-key",
    )

    assert config.get_google_api_key() == "test-key"


def test_gemini_api_key_alias():
    config = AppConfig(
        _env_file=None,
        GEMINI_API_KEY="test-key",
    )

    assert config.get_google_api_key() == "test-key"


def test_api_keys_are_optional_during_setup():
    config = AppConfig(_env_file=None)

    assert config.google_api_key is None
    assert config.openai_api_key is None
