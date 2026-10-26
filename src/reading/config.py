
from functools import lru_cache

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    google_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    )
    openai_api_key: SecretStr | None = None

    gemini_model: str = "gemini-3.8-flash"
    openai_model: str = "gpt-4o-mini"

    embed_model: str = "sentence-transformers/paraphrase-MiniLM-L3-v2"
    rerank_model: str = "BAAI/bge-reranker-base"

    direct_top_k: int = 3
    sentence_window_top_k: int = 6
    sentence_window_size: int = 3
    auto_merging_top_k: int = 12
    rerank_top_n: int = 3
    enable_reranker: bool = True

    data_dir: str = "data/documents"
    index_dir: str = "data/indexes"
    eval_dataset: str = "eval/questions.json"

    def get_google_api_key(self) -> str:
        if self.google_api_key is None:
            raise ValueError(
                "Missing GOOGLE_API_KEY or GEMINI_API_KEY in .env."
            )
        return self.google_api_key.get_secret_value()

    def get_openai_api_key(self) -> str:
        if self.openai_api_key is None:
            raise ValueError("Missing OPENAI_API_KEY in .env.")
        return self.openai_api_key.get_secret_value()


@lru_cache(maxsize=1)
def get_settings() -> AppConfig:
    return AppConfig()
