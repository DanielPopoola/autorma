from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    MODEL_SERVICE_URL: str
    MLFLOW_TRACKING_URI: str
    model_config = SettingsConfigDict(
        env_file=".env", case_sensitive=False, extra="ignore"
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
