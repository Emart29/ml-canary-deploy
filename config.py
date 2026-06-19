from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    POSTGRES_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/ml_platform"
    REDIS_URL: str = "redis://localhost:6379/2"
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_BUCKET: str = "canary-models"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8001
    HEALTH_CHECK_INTERVAL_SECONDS: int = 30
    AUTO_ROLLBACK_ENABLED: bool = True
    MAX_ERROR_RATE_DELTA: float = 0.05
    MAX_LATENCY_P95_DELTA_MS: float = 100.0
    LOG_LEVEL: str = "INFO"


settings = Settings()
