"""Cache service for the Book Recommender System using Redis."""

import logging
import json
import asyncio
from typing import Any, Dict, List, Optional, Union
from datetime import timedelta

import redis.asyncio as redis
from redis.asyncio import Redis

from book_recommender.core.config import get_settings
from book_recommender.core.exceptions import ExternalServiceError
from book_recommender.utils.helpers import pydantic_to_dict

logger = logging.getLogger(__name__)


class CacheService:
    """Redis-based cache service for storing recommendation results."""

    def __init__(self):
        self.settings = get_settings()
        self.redis_client: Optional[Redis] = None
        self.default_ttl = self.settings.redis_ttl
        self.is_enabled = self.settings.cache_enabled

    async def connect(self) -> None:
        """Connect to Redis."""
        if not self.is_enabled:
            logger.info("Cache is disabled, skipping Redis connection")
            return

        try:
            self.redis_client = redis.from_url(
                self.settings.redis_url,
                encoding='utf-8',
                decode_responses=True,
                max_connections=10
            )

            # Test connection
            await self.redis_client.ping()
            logger.info("Connected to Redis cache")

        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis_client = None
            # Don't raise exception - continue without cache

    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self.redis_client:
            await self.redis_client.close()
            self.redis_client = None
            logger.info("Disconnected from Redis cache")

    def is_connected(self) -> bool:
        """Check if Redis is connected."""
        return self.redis_client is not None and self.is_enabled

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if not self.is_connected():
            return None

        try:
            value = await self.redis_client.get(key)
            if value:
                return json.loads(value)
            return None

        except Exception as e:
            logger.warning(f"Cache get failed for key {key}: {e}")
            return None

    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None
    ) -> bool:
        """Set value in cache."""
        if not self.is_connected():
            return False

        try:
            ttl = ttl or self.default_ttl
            json_value = json.dumps(value, default=str)  # default=str handles datetime objects

            await self.redis_client.setex(key, ttl, json_value)
            return True

        except Exception as e:
            logger.warning(f"Cache set failed for key {key}: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        if not self.is_connected():
            return False

        try:
            result = await self.redis_client.delete(key)
            return result > 0

        except Exception as e:
            logger.warning(f"Cache delete failed for key {key}: {e}")
            return False

    async def delete_pattern(self, pattern: str) -> int:
        """Delete keys matching pattern."""
        if not self.is_connected():
            return 0

        try:
            keys = await self.redis_client.keys(pattern)
            if keys:
                result = await self.redis_client.delete(*keys)
                return result
            return 0

        except Exception as e:
            logger.warning(f"Cache delete pattern failed for pattern {pattern}: {e}")
            return 0

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        if not self.is_connected():
            return False

        try:
            result = await self.redis_client.exists(key)
            return result > 0

        except Exception as e:
            logger.warning(f"Cache exists check failed for key {key}: {e}")
            return False

    async def expire(self, key: str, ttl: int) -> bool:
        """Set expiration time for a key."""
        if not self.is_connected():
            return False

        try:
            result = await self.redis_client.expire(key, ttl)
            return result

        except Exception as e:
            logger.warning(f"Cache expire failed for key {key}: {e}")
            return False

    async def ttl(self, key: str) -> int:
        """Get TTL for a key."""
        if not self.is_connected():
            return -1

        try:
            result = await self.redis_client.ttl(key)
            return result

        except Exception as e:
            logger.warning(f"Cache TTL check failed for key {key}: {e}")
            return -1

    async def mget(self, keys: List[str]) -> List[Optional[Any]]:
        """Get multiple values from cache."""
        if not self.is_connected():
            return [None] * len(keys)

        try:
            values = await self.redis_client.mget(keys)
            results = []

            for value in values:
                if value:
                    try:
                        results.append(json.loads(value))
                    except json.JSONDecodeError:
                        results.append(None)
                else:
                    results.append(None)

            return results

        except Exception as e:
            logger.warning(f"Cache mget failed: {e}")
            return [None] * len(keys)

    async def mset(self, mapping: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set multiple key-value pairs."""
        if not self.is_connected():
            return False

        try:
            # Convert values to JSON
            json_mapping = {}
            for key, value in mapping.items():
                json_mapping[key] = json.dumps(value, default=str)

            # Set all keys
            await self.redis_client.mset(json_mapping)

            # Set expiration if provided
            if ttl:
                ttl = ttl or self.default_ttl
                pipe = self.redis_client.pipeline()
                for key in mapping.keys():
                    pipe.expire(key, ttl)
                await pipe.execute()

            return True

        except Exception as e:
            logger.warning(f"Cache mset failed: {e}")
            return False

    async def incr(self, key: str, amount: int = 1) -> Optional[int]:
        """Increment a counter."""
        if not self.is_connected():
            return None

        try:
            result = await self.redis_client.incr(key, amount)
            return result

        except Exception as e:
            logger.warning(f"Cache incr failed for key {key}: {e}")
            return None

    async def clear_all(self) -> int:
        """Clear all keys from cache. Use with caution!"""
        if not self.is_connected():
            return 0

        try:
            # Get all keys
            keys = await self.redis_client.keys("*")
            if keys:
                result = await self.redis_client.delete(*keys)
                return result
            return 0

        except Exception as e:
            logger.error(f"Cache clear all failed: {e}")
            return 0

    async def get_info(self) -> Dict[str, Any]:
        """Get cache information and statistics."""
        if not self.is_connected():
            return {"status": "disconnected"}

        try:
            info = await self.redis_client.info()
            memory_info = await self.redis_client.info("memory")

            return {
                "status": "connected",
                "redis_version": info.get("redis_version"),
                "used_memory_human": memory_info.get("used_memory_human"),
                "used_memory_peak_human": memory_info.get("used_memory_peak_human"),
                "connected_clients": info.get("connected_clients"),
                "total_commands_processed": info.get("total_commands_processed"),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0)
            }

        except Exception as e:
            logger.error(f"Failed to get cache info: {e}")
            return {"status": "error", "error": str(e)}

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on cache service."""
        if not self.is_enabled:
            return {
                "status": "disabled",
                "healthy": True,
                "message": "Cache service is disabled"
            }

        try:
            if not self.redis_client:
                await self.connect()

            # Test basic operations
            test_key = "health_check_test"
            test_value = {"timestamp": "2024-01-01T00:00:00Z"}

            # Test set
            await self.set(test_key, test_value, ttl=10)

            # Test get
            retrieved_value = await self.get(test_key)

            # Test delete
            await self.delete(test_key)

            if retrieved_value == test_value:
                return {
                    "status": "healthy",
                    "healthy": True,
                    "message": "All cache operations working"
                }
            else:
                return {
                    "status": "unhealthy",
                    "healthy": False,
                    "message": "Cache data integrity issue"
                }

        except Exception as e:
            logger.error(f"Cache health check failed: {e}")
            return {
                "status": "unhealthy",
                "healthy": False,
                "message": f"Cache health check failed: {str(e)}"
            }

    def create_key(self, prefix: str, *args, **kwargs) -> str:
        """Create a standardized cache key."""
        key_parts = [prefix]

        # Add positional arguments
        for arg in args:
            key_parts.append(str(arg))

        # Add keyword arguments
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}:{v}")

        return ":".join(key_parts)


# Global cache service instance
cache_service = CacheService()


# Context manager for cache operations
class CacheContext:
    """Context manager for cache operations with automatic cleanup."""

    def __init__(self, cache_service: CacheService):
        self.cache_service = cache_service

    async def __aenter__(self):
        await self.cache_service.connect()
        return self.cache_service

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cache_service.disconnect()


# Decorator for caching function results
def cached(key_prefix: str, ttl: Optional[int] = None):
    """Decorator for caching function results."""

    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Create cache key
            cache_key = cache_service.create_key(key_prefix, *args, **kwargs)

            # Try to get from cache
            cached_result = await cache_service.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Execute function
            result = await func(*args, **kwargs)

            # Cache result
            await cache_service.set(cache_key, result, ttl)

            return result

        return wrapper
    return decorator
