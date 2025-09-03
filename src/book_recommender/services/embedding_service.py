"""Embedding service for handling text embeddings using various providers."""

import logging
import asyncio
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

import numpy as np
import aiohttp
from sklearn.feature_extraction.text import TfidfVectorizer

from book_recommender.core.config import get_settings
from book_recommender.core.exceptions import ExternalServiceError
from book_recommender.utils.helpers import retry_async, normalize_text

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    async def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a single text."""
        pass

    @abstractmethod
    async def get_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Get embeddings for multiple texts."""
        pass

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Get embedding dimension."""
        pass


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Ollama embedding provider."""

    def __init__(self, ollama_url: str, model_name: str, embedding_dim: int = 2304):
        self.ollama_url = ollama_url
        self.model_name = model_name
        self._embedding_dim = embedding_dim
        self.session = None

    async def _ensure_session(self):
        """Ensure aiohttp session exists."""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=60, connect=10)
            self.session = aiohttp.ClientSession(timeout=timeout)

    async def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a single text using Ollama."""
        await self._ensure_session()

        async def make_request():
            async with self.session.post(
                f"{self.ollama_url}/api/embeddings",
                json={
                    'model': self.model_name,
                    'prompt': normalize_text(text)
                }
            ) as response:
                if response.status != 200:
                    raise ExternalServiceError(f"Ollama API returned status {response.status}")

                data = await response.json()
                if 'embedding' not in data:
                    raise ExternalServiceError("No embedding in Ollama response")

                embedding = np.array(data['embedding'], dtype=np.float32)

                # Ensure correct dimension
                if len(embedding) != self._embedding_dim:
                    logger.warning(f"Expected embedding dimension {self._embedding_dim}, got {len(embedding)}")
                    # Pad or truncate as needed
                    if len(embedding) < self._embedding_dim:
                        padding = np.zeros(self._embedding_dim - len(embedding))
                        embedding = np.concatenate([embedding, padding])
                    else:
                        embedding = embedding[:self._embedding_dim]

                return embedding

        return await retry_async(make_request, max_retries=3, delay=1.0, exceptions=(ExternalServiceError, aiohttp.ClientError))

    async def get_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Get embeddings for multiple texts."""
        embeddings = []

        # Process in batches to avoid overwhelming the API
        batch_size = 10
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # Process batch concurrently
            tasks = [self.get_embedding(text) for text in batch_texts]
            batch_embeddings = await asyncio.gather(*tasks, return_exceptions=True)

            for embedding in batch_embeddings:
                if isinstance(embedding, Exception):
                    logger.warning(f"Failed to get embedding: {embedding}")
                    # Use zero embedding as fallback
                    embeddings.append(np.zeros(self._embedding_dim, dtype=np.float32))
                else:
                    embeddings.append(embedding)

            # Rate limiting
            if i + batch_size < len(texts):
                await asyncio.sleep(0.1)  # Small delay between batches

        return embeddings

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    async def close(self):
        """Close the aiohttp session."""
        if self.session:
            await self.session.close()
            self.session = None


class TFIDFEmbeddingProvider(EmbeddingProvider):
    """TF-IDF based embedding provider as fallback."""

    def __init__(self, max_features: int = 5000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        self.is_fitted = False
        self._embedding_dim = max_features

    def fit(self, texts: List[str]):
        """Fit the TF-IDF vectorizer on corpus."""
        cleaned_texts = [normalize_text(text) for text in texts]
        self.vectorizer.fit(cleaned_texts)
        self.is_fitted = True
        self._embedding_dim = len(self.vectorizer.get_feature_names_out())

    async def get_embedding(self, text: str) -> np.ndarray:
        """Get TF-IDF embedding for a single text."""
        if not self.is_fitted:
            raise RuntimeError("TF-IDF vectorizer is not fitted")

        cleaned_text = normalize_text(text)
        vector = self.vectorizer.transform([cleaned_text])
        return vector.toarray()[0].astype(np.float32)

    async def get_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Get TF-IDF embeddings for multiple texts."""
        if not self.is_fitted:
            raise RuntimeError("TF-IDF vectorizer is not fitted")

        cleaned_texts = [normalize_text(text) for text in texts]
        vectors = self.vectorizer.transform(cleaned_texts)
        return [vector.astype(np.float32) for vector in vectors.toarray()]

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim


class EmbeddingService:
    """Service for managing text embeddings using different providers."""

    def __init__(self):
        self.settings = get_settings()
        self.primary_provider: Optional[EmbeddingProvider] = None
        self.fallback_provider: Optional[EmbeddingProvider] = None
        self.is_initialized = False

    async def initialize(self, corpus_texts: Optional[List[str]] = None):
        """Initialize embedding providers."""
        try:
            # Initialize Ollama provider as primary
            self.primary_provider = OllamaEmbeddingProvider(
                ollama_url=self.settings.ollama_url,
                model_name=self.settings.ollama_model,
                embedding_dim=self.settings.models.content_based.embedding_dim
            )

            # Test Ollama connection
            test_embedding = await self.primary_provider.get_embedding("test")
            logger.info(f"Ollama provider initialized successfully, embedding dim: {len(test_embedding)}")

        except Exception as e:
            logger.warning(f"Failed to initialize Ollama provider: {e}")
            self.primary_provider = None

        # Initialize TF-IDF fallback provider
        self.fallback_provider = TFIDFEmbeddingProvider()

        if corpus_texts:
            self.fallback_provider.fit(corpus_texts)
            logger.info("TF-IDF fallback provider fitted on corpus")

        self.is_initialized = True
        logger.info("Embedding service initialized")

    async def get_embedding(self, text: str, use_fallback_on_error: bool = True) -> np.ndarray:
        """Get embedding for a single text."""
        if not self.is_initialized:
            raise RuntimeError("Embedding service is not initialized")

        # Try primary provider first
        if self.primary_provider:
            try:
                return await self.primary_provider.get_embedding(text)
            except Exception as e:
                logger.warning(f"Primary embedding provider failed: {e}")
                if not use_fallback_on_error:
                    raise

        # Fall back to TF-IDF
        if self.fallback_provider and self.fallback_provider.is_fitted:
            try:
                return await self.fallback_provider.get_embedding(text)
            except Exception as e:
                logger.error(f"Fallback embedding provider failed: {e}")
                raise

        raise RuntimeError("No embedding provider available")

    async def get_embeddings(
        self, 
        texts: List[str], 
        use_fallback_on_error: bool = True,
        show_progress: bool = False
    ) -> List[np.ndarray]:
        """Get embeddings for multiple texts."""
        if not self.is_initialized:
            raise RuntimeError("Embedding service is not initialized")

        # Try primary provider first
        if self.primary_provider:
            try:
                if show_progress:
                    logger.info(f"Getting embeddings for {len(texts)} texts using primary provider")

                embeddings = await self.primary_provider.get_embeddings(texts)

                if show_progress:
                    logger.info("Successfully got embeddings using primary provider")

                return embeddings

            except Exception as e:
                logger.warning(f"Primary embedding provider failed for batch: {e}")
                if not use_fallback_on_error:
                    raise

        # Fall back to TF-IDF
        if self.fallback_provider and self.fallback_provider.is_fitted:
            try:
                if show_progress:
                    logger.info(f"Getting embeddings for {len(texts)} texts using fallback provider")

                embeddings = await self.fallback_provider.get_embeddings(texts)

                if show_progress:
                    logger.info("Successfully got embeddings using fallback provider")

                return embeddings

            except Exception as e:
                logger.error(f"Fallback embedding provider failed for batch: {e}")
                raise

        raise RuntimeError("No embedding provider available")

    def get_embedding_dim(self) -> int:
        """Get the embedding dimension."""
        if self.primary_provider:
            return self.primary_provider.embedding_dim
        elif self.fallback_provider:
            return self.fallback_provider.embedding_dim
        else:
            return self.settings.models.content_based.embedding_dim

    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about available providers."""
        info = {
            'primary_provider': None,
            'fallback_provider': None,
            'embedding_dim': self.get_embedding_dim()
        }

        if self.primary_provider:
            info['primary_provider'] = {
                'type': 'ollama',
                'model': self.settings.ollama_model,
                'url': self.settings.ollama_url,
                'dimension': self.primary_provider.embedding_dim
            }

        if self.fallback_provider and self.fallback_provider.is_fitted:
            info['fallback_provider'] = {
                'type': 'tfidf',
                'dimension': self.fallback_provider.embedding_dim,
                'fitted': True
            }

        return info

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on embedding service."""
        health_info = {
            'status': 'healthy' if self.is_initialized else 'not_initialized',
            'providers': {}
        }

        # Test primary provider
        if self.primary_provider:
            try:
                test_embedding = await self.primary_provider.get_embedding("health check test")
                health_info['providers']['primary'] = {
                    'status': 'healthy',
                    'type': 'ollama',
                    'dimension': len(test_embedding)
                }
            except Exception as e:
                health_info['providers']['primary'] = {
                    'status': 'unhealthy',
                    'type': 'ollama',
                    'error': str(e)
                }

        # Test fallback provider
        if self.fallback_provider and self.fallback_provider.is_fitted:
            try:
                test_embedding = await self.fallback_provider.get_embedding("health check test")
                health_info['providers']['fallback'] = {
                    'status': 'healthy',
                    'type': 'tfidf',
                    'dimension': len(test_embedding)
                }
            except Exception as e:
                health_info['providers']['fallback'] = {
                    'status': 'unhealthy',
                    'type': 'tfidf',
                    'error': str(e)
                }

        # Determine overall health
        healthy_providers = sum(1 for p in health_info['providers'].values() 
                              if p.get('status') == 'healthy')

        if healthy_providers == 0:
            health_info['status'] = 'unhealthy'
        elif healthy_providers == 1:
            health_info['status'] = 'degraded'
        else:
            health_info['status'] = 'healthy'

        return health_info

    async def close(self):
        """Close the embedding service and cleanup resources."""
        if self.primary_provider and hasattr(self.primary_provider, 'close'):
            await self.primary_provider.close()

        logger.info("Embedding service closed")


# Global embedding service instance
embedding_service = EmbeddingService()
