"""Configuration management for the Book Recommender System."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class AppConfig(BaseModel):
    """Application configuration."""
    name: str = "Book Recommender System"
    version: str = "1.0.0"
    description: str = "A production-ready book recommendation system"
    debug: bool = False


class ServerConfig(BaseModel):
    """Server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    workers: int = 4


class DatabaseConfig(BaseModel):
    """Database configuration."""
    url: str = "sqlite:///./book_recommender.db"
    echo: bool = False
    pool_size: int = 10
    max_overflow: int = 20


class RedisConfig(BaseModel):
    """Redis configuration."""
    url: str = "redis://localhost:6379/0"
    ttl: int = 3600
    max_connections: int = 10


class CollaborativeFilteringConfig(BaseModel):
    """Collaborative filtering model configuration."""
    algorithm: str = "brute"
    n_neighbors: int = 6
    min_user_ratings: int = 200
    min_book_ratings: int = 50


class ContentBasedConfig(BaseModel):
    """Content-based model configuration."""
    embedding_dim: int = 2304
    similarity_metric: str = "cosine"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "gemma2:2b"


class HybridConfig(BaseModel):
    """Hybrid model configuration."""
    collaborative_weight: float = 0.6
    content_weight: float = 0.4


class ModelsConfig(BaseModel):
    """Models configuration."""
    collaborative_filtering: CollaborativeFilteringConfig = CollaborativeFilteringConfig()
    content_based: ContentBasedConfig = ContentBasedConfig()
    hybrid: HybridConfig = HybridConfig()


class SecurityConfig(BaseModel):
    """Security configuration."""
    secret_key: str = "change-me-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30


class RateLimitingConfig(BaseModel):
    """Rate limiting configuration."""
    requests_per_minute: int = 100
    burst_size: int = 10


class PerformanceConfig(BaseModel):
    """Performance configuration."""
    cache_enabled: bool = True
    batch_size: int = 1000
    max_recommendations: int = 20
    similarity_threshold: float = 0.5


class MonitoringConfig(BaseModel):
    """Monitoring configuration."""
    enable_metrics: bool = True
    health_check_interval: int = 30
    log_level: str = "INFO"


class DataConfig(BaseModel):
    """Data configuration."""
    raw_data_path: str = "data/raw"
    processed_data_path: str = "data/processed"
    models_path: str = "data/models"


class Settings(BaseSettings):
    """Main application settings."""

    # Environment
    environment: str = Field(default="development", env="ENVIRONMENT")

    # App configuration
    app_name: str = Field(default="Book Recommender System", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    app_description: str = Field(default="A production-ready book recommendation system", env="APP_DESCRIPTION")
    debug: bool = Field(default=False, env="DEBUG")

    # Server configuration
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    reload: bool = Field(default=False, env="RELOAD")
    workers: int = Field(default=4, env="WORKERS")

    # Database
    database_url: str = Field(default="sqlite:///./book_recommender.db", env="DATABASE_URL")
    database_echo: bool = Field(default=False, env="DATABASE_ECHO")

    # Redis
    redis_url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    redis_ttl: int = Field(default=3600, env="REDIS_TTL")

    # Security
    secret_key: str = Field(default="change-me-in-production", env="SECRET_KEY")
    algorithm: str = Field(default="HS256", env="ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")

    # Ollama
    ollama_url: str = Field(default="http://localhost:11434", env="OLLAMA_URL")
    ollama_model: str = Field(default="gemma2:2b", env="OLLAMA_MODEL")

    # Performance
    cache_enabled: bool = Field(default=True, env="CACHE_ENABLED")
    batch_size: int = Field(default=1000, env="BATCH_SIZE")
    max_recommendations: int = Field(default=20, env="MAX_RECOMMENDATIONS")

    # Monitoring
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")

    # Rate limiting
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=60, env="RATE_LIMIT_WINDOW")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


def load_config_file(config_path: Path) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if not config_path.exists():
        return {}

    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge configuration dictionaries."""
    result = base_config.copy()

    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value

    return result


@lru_cache()
def get_settings() -> Settings:
    """Get application settings with caching."""
    # Load base configuration
    config_dir = Path("config")
    base_config = load_config_file(config_dir / "config.yaml")

    # Load environment-specific configuration
    environment = os.getenv("ENVIRONMENT", "development")
    env_config = load_config_file(config_dir / f"config.{environment}.yaml")

    # Merge configurations
    merged_config = merge_configs(base_config, env_config)

    return Settings(**merged_config)


# Global settings instance
settings = get_settings()
