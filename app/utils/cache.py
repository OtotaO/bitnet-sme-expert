"""Simple in-memory cache implementation."""

import hashlib
import logging
from datetime import UTC, datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)


class SimpleCache:
    """A simple in-memory cache with TTL support."""

    def __init__(self, ttl_seconds: int = 300, max_size: int = 1000):
        """Initialize the cache.

        Args:
            ttl_seconds: Time to live for cache entries in seconds
            max_size: Maximum number of items to store in the cache
        """
        self._cache: dict[str, dict] = {}
        self.ttl = ttl_seconds
        self.max_size = max_size
        logger.info(f"Cache initialized with TTL={ttl_seconds}s, max_size={max_size}")

    def _get_key(self, *args, **kwargs) -> str:
        """Generate a cache key from function arguments."""
        key_parts = [str(arg) for arg in args]
        key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
        key_string = ":".join(key_parts)
        return hashlib.md5(key_string.encode(), usedforsecurity=False).hexdigest()

    def get(self, key: str) -> Any | None:
        """Get a value from the cache."""
        if key not in self._cache:
            return None

        entry = self._cache[key]
        if datetime.now(UTC) > entry["expires_at"]:
            del self._cache[key]
            return None

        return entry["value"]

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set a value in the cache."""
        # Evict expired entries if we're approaching the size limit
        if len(self._cache) >= self.max_size:
            self._evict_expired()

        # If still full, remove the oldest entry
        if len(self._cache) >= self.max_size and self._cache:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            logger.debug(f"Evicted oldest cache entry: {oldest_key}")

        ttl = ttl or self.ttl
        self._cache[key] = {
            "value": value,
            "expires_at": datetime.now(UTC) + timedelta(seconds=ttl),
            "created_at": datetime.now(UTC),
        }

    def _evict_expired(self) -> None:
        """Remove all expired entries from the cache."""
        now = datetime.now(UTC)
        expired_keys = [k for k, v in self._cache.items() if v["expires_at"] < now]
        for key in expired_keys:
            del self._cache[key]

        if expired_keys:
            logger.debug(f"Evicted {len(expired_keys)} expired cache entries")

    def clear(self) -> None:
        """Clear the entire cache."""
        self._cache.clear()
        logger.info("Cache cleared")

    def stats(self) -> dict:
        """Get cache statistics."""
        now = datetime.now(UTC)
        expired = sum(1 for v in self._cache.values() if v["expires_at"] < now)

        return {
            "total_entries": len(self._cache),
            "expired_entries": expired,
            "max_size": self.max_size,
            "ttl_seconds": self.ttl,
        }


# Global cache instance
cache = SimpleCache(ttl_seconds=300, max_size=1000)
