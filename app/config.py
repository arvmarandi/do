from pydantic_settings import BaseSettings, SettingsConfigDict

# config that reads from a .env file
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    worker_pool_size: int = 10
    max_retries: int = 3
    inference_url: str = "http://localhost:8081/infer"
    db_path: str = "batches.db"


settings = Settings()
