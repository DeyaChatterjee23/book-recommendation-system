"""Recommendation service for orchestrating different recommendation models."""

import logging
from typing import Dict, List, Optional, Union
import asyncio

from book_recommender.core.config import get_settings
from book_recommender.core.exceptions import ModelNotLoadedError, ValidationError
from book_recommender.models.collaborative_filtering import collaborative_model
from book_recommender.models.content_based import content_model
from book_recommender.models.hybrid_model import hybrid_model
from book_recommender.services.cache_service import cache_service
from book_recommender.data.data_validator import data_validator
from book_recommender.utils.constants import ModelTypes, CacheKeys

logger = logging.getLogger(__name__)


class RecommendationService:
    """Service for managing and orchestrating recommendation models."""

    def __init__(self):
        self.settings = get_settings()
        self.cache_service = cache_service
        self.data_validator = data_validator

        # Model instances
        self.collaborative_model = collaborative_model
        self.content_model = content_model
        self.hybrid_model = hybrid_model

    async def get_collaborative_recommendations(
        self, 
        user_id: str, 
        num_recommendations: int = 10
    ) -> Dict[str, any]:
        """Get collaborative filtering recommendations for a user."""

        # Validate input
        validation_result = self.data_validator.validate_model_input(
            {'user_id': user_id, 'num_recommendations': num_recommendations},
            ModelTypes.COLLABORATIVE_FILTERING
        )

        if not validation_result['valid']:
            raise ValidationError(f"Invalid input: {validation_result['errors']}")

        # Check cache first
        cache_key = CacheKeys.USER_RECOMMENDATIONS.format(user_id=user_id)
        cached_result = await self.cache_service.get(f"{cache_key}_{num_recommendations}")

        if cached_result:
            logger.info(f"Returning cached collaborative recommendations for user {user_id}")
            return cached_result

        try:
            # Generate recommendations
            recommendations = self.collaborative_model.predict_for_user(user_id, num_recommendations)

            result = {
                'user_id': user_id,
                'model_type': ModelTypes.COLLABORATIVE_FILTERING,
                'recommendations': recommendations,
                'total_count': len(recommendations)
            }

            # Cache the result
            await self.cache_service.set(
                f"{cache_key}_{num_recommendations}", 
                result, 
                ttl=self.settings.redis_ttl
            )

            logger.info(f"Generated {len(recommendations)} collaborative recommendations for user {user_id}")
            return result

        except ModelNotLoadedError:
            raise ModelNotLoadedError("Collaborative filtering model is not available")
        except Exception as e:
            logger.error(f"Failed to generate collaborative recommendations: {e}")
            raise

    async def get_content_based_recommendations(
        self, 
        book_id: int, 
        num_recommendations: int = 10
    ) -> Dict[str, any]:
        """Get content-based recommendations for a book."""

        # Validate input
        validation_result = self.data_validator.validate_model_input(
            {'book_id': book_id, 'num_recommendations': num_recommendations},
            ModelTypes.CONTENT_BASED
        )

        if not validation_result['valid']:
            raise ValidationError(f"Invalid input: {validation_result['errors']}")

        # Check cache first
        cache_key = CacheKeys.BOOK_RECOMMENDATIONS.format(book_id=book_id)
        cached_result = await self.cache_service.get(f"{cache_key}_{num_recommendations}")

        if cached_result:
            logger.info(f"Returning cached content recommendations for book {book_id}")
            return cached_result

        try:
            # Generate recommendations
            recommendations = self.content_model.predict(book_id, num_recommendations)

            # Get book details
            book_details = self.content_model.get_book_details(book_id)

            result = {
                'book_id': book_id,
                'book_details': book_details,
                'model_type': ModelTypes.CONTENT_BASED,
                'recommendations': recommendations,
                'total_count': len(recommendations)
            }

            # Cache the result
            await self.cache_service.set(
                f"{cache_key}_{num_recommendations}", 
                result, 
                ttl=self.settings.redis_ttl
            )

            logger.info(f"Generated {len(recommendations)} content recommendations for book {book_id}")
            return result

        except ModelNotLoadedError:
            raise ModelNotLoadedError("Content-based model is not available")
        except Exception as e:
            logger.error(f"Failed to generate content recommendations: {e}")
            raise

    async def get_hybrid_recommendations(
        self,
        user_id: Optional[str] = None,
        book_id: Optional[int] = None,
        preferences: Optional[Dict[str, any]] = None,
        num_recommendations: int = 10
    ) -> Dict[str, any]:
        """Get hybrid recommendations."""

        # Validate input
        input_data = {
            'user_id': user_id,
            'book_id': book_id,
            'preferences': preferences,
            'num_recommendations': num_recommendations
        }

        validation_result = self.data_validator.validate_model_input(
            input_data, ModelTypes.HYBRID
        )

        if not validation_result['valid']:
            raise ValidationError(f"Invalid input: {validation_result['errors']}")

        # Create cache key based on input
        cache_key_components = [f"hybrid_{num_recommendations}"]
        if user_id:
            cache_key_components.append(f"user_{user_id}")
        if book_id:
            cache_key_components.append(f"book_{book_id}")
        if preferences:
            # Create a simple hash of preferences for caching
            import hashlib
            pref_str = str(sorted(preferences.items()))
            pref_hash = hashlib.md5(pref_str.encode()).hexdigest()[:8]
            cache_key_components.append(f"pref_{pref_hash}")

        cache_key = "_".join(cache_key_components)

        # Check cache
        cached_result = await self.cache_service.get(cache_key)
        if cached_result:
            logger.info("Returning cached hybrid recommendations")
            return cached_result

        try:
            # Generate recommendations
            recommendations = await self.hybrid_model.predict(
                user_id=user_id,
                book_id=book_id,
                preferences=preferences,
                num_recommendations=num_recommendations
            )

            result = {
                'model_type': ModelTypes.HYBRID,
                'input_data': {
                    'user_id': user_id,
                    'book_id': book_id,
                    'has_preferences': preferences is not None
                },
                'recommendations': recommendations,
                'total_count': len(recommendations)
            }

            # Cache the result
            await self.cache_service.set(cache_key, result, ttl=self.settings.redis_ttl)

            logger.info(f"Generated {len(recommendations)} hybrid recommendations")
            return result

        except ModelNotLoadedError:
            raise ModelNotLoadedError("Hybrid model is not available")
        except Exception as e:
            logger.error(f"Failed to generate hybrid recommendations: {e}")
            raise

    async def search_books(self, query: str, num_results: int = 20) -> Dict[str, any]:
        """Search for books by text query."""

        if not query or len(query.strip()) == 0:
            raise ValidationError("Search query cannot be empty")

        if len(query) > 1000:
            raise ValidationError("Search query too long")

        # Check cache
        cache_key = f"search_{query}_{num_results}"
        cached_result = await self.cache_service.get(cache_key)

        if cached_result:
            logger.info(f"Returning cached search results for query: {query}")
            return cached_result

        try:
            # Use content-based model for search
            results = self.content_model.search_books(query, num_results)

            result = {
                'query': query,
                'results': results,
                'total_count': len(results)
            }

            # Cache the result
            await self.cache_service.set(cache_key, result, ttl=self.settings.redis_ttl)

            logger.info(f"Found {len(results)} books for query: {query}")
            return result

        except ModelNotLoadedError:
            raise ModelNotLoadedError("Search functionality is not available")
        except Exception as e:
            logger.error(f"Failed to search books: {e}")
            raise

    async def get_book_details(self, book_id: int) -> Dict[str, any]:
        """Get detailed information about a book."""

        # Check cache
        cache_key = CacheKeys.BOOK_DETAILS.format(book_id=book_id)
        cached_result = await self.cache_service.get(cache_key)

        if cached_result:
            logger.info(f"Returning cached book details for book {book_id}")
            return cached_result

        try:
            # Get book details from content model
            book_details = self.content_model.get_book_details(book_id)

            # Cache the result
            await self.cache_service.set(cache_key, book_details, ttl=self.settings.redis_ttl * 2)  # Cache longer for book details

            logger.info(f"Retrieved details for book {book_id}")
            return book_details

        except ModelNotLoadedError:
            raise ModelNotLoadedError("Book details service is not available")
        except Exception as e:
            logger.error(f"Failed to get book details: {e}")
            raise

    async def get_similar_books(self, book_title: str, num_recommendations: int = 10) -> Dict[str, any]:
        """Get books similar to a given book title."""

        if not book_title or len(book_title.strip()) == 0:
            raise ValidationError("Book title cannot be empty")

        # Check cache
        cache_key = f"similar_{book_title}_{num_recommendations}"
        cached_result = await self.cache_service.get(cache_key)

        if cached_result:
            logger.info(f"Returning cached similar books for: {book_title}")
            return cached_result

        try:
            # Use hybrid model for better results
            recommendations = await self.hybrid_model.predict_similar_books(book_title, num_recommendations)

            result = {
                'book_title': book_title,
                'model_type': 'hybrid_similarity',
                'recommendations': recommendations,
                'total_count': len(recommendations)
            }

            # Cache the result
            await self.cache_service.set(cache_key, result, ttl=self.settings.redis_ttl)

            logger.info(f"Found {len(recommendations)} similar books for: {book_title}")
            return result

        except Exception as e:
            logger.error(f"Failed to find similar books: {e}")
            # Fallback to collaborative filtering
            try:
                recommendations = self.collaborative_model.predict(book_title, num_recommendations)
                result = {
                    'book_title': book_title,
                    'model_type': ModelTypes.COLLABORATIVE_FILTERING,
                    'recommendations': recommendations,
                    'total_count': len(recommendations)
                }
                return result
            except Exception as e2:
                logger.error(f"Fallback also failed: {e2}")
                raise

    async def get_popular_books(self, num_books: int = 20) -> Dict[str, any]:
        """Get popular books."""

        # Check cache
        cache_key = f"{CacheKeys.POPULAR_BOOKS}_{num_books}"
        cached_result = await self.cache_service.get(cache_key)

        if cached_result:
            logger.info("Returning cached popular books")
            return cached_result

        try:
            # Use collaborative model to get popular books
            popular_books = self.collaborative_model._get_popular_books(num_books)

            result = {
                'model_type': 'popularity',
                'books': popular_books,
                'total_count': len(popular_books)
            }

            # Cache the result for longer time (popular books don't change frequently)
            await self.cache_service.set(cache_key, result, ttl=self.settings.redis_ttl * 6)

            logger.info(f"Retrieved {len(popular_books)} popular books")
            return result

        except Exception as e:
            logger.error(f"Failed to get popular books: {e}")
            raise

    def get_model_status(self) -> Dict[str, any]:
        """Get status of all recommendation models."""

        status = {
            'collaborative_filtering': self.collaborative_model.get_model_info(),
            'content_based': self.content_model.get_model_info(),
            'hybrid': self.hybrid_model.get_model_info(),
            'cache_service': {
                'status': 'available' if self.cache_service.is_connected() else 'unavailable'
            }
        }

        # Overall system status
        models_ready = sum(1 for model_info in status.values() 
                          if isinstance(model_info, dict) and model_info.get('status') == 'trained')

        status['system'] = {
            'models_ready': models_ready,
            'total_models': 3,
            'ready_percentage': (models_ready / 3) * 100
        }

        return status

    async def clear_cache(self, pattern: Optional[str] = None) -> Dict[str, any]:
        """Clear recommendation cache."""

        try:
            if pattern:
                cleared_count = await self.cache_service.delete_pattern(pattern)
            else:
                cleared_count = await self.cache_service.clear_all()

            result = {
                'status': 'success',
                'cleared_keys': cleared_count,
                'pattern': pattern
            }

            logger.info(f"Cleared {cleared_count} cache keys")
            return result

        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            raise

    async def warmup_cache(self, sample_requests: List[Dict[str, any]]) -> Dict[str, any]:
        """Warm up cache with sample requests."""

        logger.info(f"Warming up cache with {len(sample_requests)} requests")

        successful_requests = 0
        failed_requests = 0

        for request in sample_requests:
            try:
                request_type = request.get('type')

                if request_type == 'collaborative':
                    await self.get_collaborative_recommendations(
                        request['user_id'], 
                        request.get('num_recommendations', 10)
                    )
                elif request_type == 'content':
                    await self.get_content_based_recommendations(
                        request['book_id'], 
                        request.get('num_recommendations', 10)
                    )
                elif request_type == 'hybrid':
                    await self.get_hybrid_recommendations(
                        user_id=request.get('user_id'),
                        book_id=request.get('book_id'),
                        preferences=request.get('preferences'),
                        num_recommendations=request.get('num_recommendations', 10)
                    )
                elif request_type == 'search':
                    await self.search_books(
                        request['query'], 
                        request.get('num_results', 20)
                    )

                successful_requests += 1

            except Exception as e:
                logger.warning(f"Cache warmup request failed: {e}")
                failed_requests += 1

        result = {
            'status': 'completed',
            'total_requests': len(sample_requests),
            'successful': successful_requests,
            'failed': failed_requests
        }

        logger.info(f"Cache warmup completed: {result}")
        return result


# Global recommendation service instance
recommendation_service = RecommendationService()
