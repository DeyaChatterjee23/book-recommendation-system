"""Hybrid recommendation model combining collaborative and content-based filtering."""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from book_recommender.core.config import get_settings
from book_recommender.core.exceptions import ModelNotLoadedError, DataNotFoundError
from book_recommender.models.collaborative_filtering import CollaborativeFilteringModel
from book_recommender.models.content_based import ContentBasedModel

logger = logging.getLogger(__name__)
settings = get_settings()


class HybridModel:
    """Hybrid recommendation model combining collaborative and content-based approaches."""

    def __init__(self):
        self.settings = get_settings()
        self.collaborative_model = CollaborativeFilteringModel()
        self.content_model = ContentBasedModel()

        # Hybrid weights
        self.collaborative_weight = self.settings.models.hybrid.collaborative_weight
        self.content_weight = self.settings.models.hybrid.content_weight

        # Ensure weights sum to 1
        total_weight = self.collaborative_weight + self.content_weight
        self.collaborative_weight /= total_weight
        self.content_weight /= total_weight

        self.is_trained = False

    async def train(
        self, 
        ratings_df: pd.DataFrame, 
        books_df: pd.DataFrame, 
        content_books_df: pd.DataFrame,
        use_ollama: bool = True
    ) -> Dict[str, any]:
        """Train both collaborative and content-based models."""
        logger.info("Training Hybrid model")

        try:
            # Train collaborative filtering model
            logger.info("Training collaborative filtering component")
            collab_stats = self.collaborative_model.train(ratings_df, books_df)

            # Train content-based model
            logger.info("Training content-based component")
            content_stats = await self.content_model.train(content_books_df, use_ollama)

            self.is_trained = True

            # Combined training statistics
            training_stats = {
                'model_type': 'hybrid',
                'collaborative_weight': self.collaborative_weight,
                'content_weight': self.content_weight,
                'collaborative_stats': collab_stats,
                'content_stats': content_stats
            }

            logger.info(f"Hybrid model training completed: {training_stats}")
            return training_stats

        except Exception as e:
            logger.error(f"Failed to train hybrid model: {e}")
            raise

    def predict(
        self, 
        user_id: Optional[str] = None,
        book_id: Optional[int] = None,
        preferences: Optional[Dict[str, any]] = None,
        num_recommendations: int = 10
    ) -> List[Dict[str, any]]:
        """Generate hybrid recommendations."""
        if not self.is_trained:
            raise ModelNotLoadedError("Hybrid model is not trained")

        try:
            logger.info(f"Generating hybrid recommendations for user_id={user_id}, book_id={book_id}")

            collaborative_recommendations = []
            content_recommendations = []

            # Get collaborative recommendations
            if user_id is not None and self.collaborative_model.is_trained:
                try:
                    collaborative_recommendations = self.collaborative_model.predict_for_user(
                        user_id, num_recommendations * 2  # Get more to have options for merging
                    )
                    logger.info(f"Got {len(collaborative_recommendations)} collaborative recommendations")
                except Exception as e:
                    logger.warning(f"Collaborative recommendations failed: {e}")

            # Get content-based recommendations
            if self.content_model.is_trained:
                try:
                    if book_id is not None:
                        content_recommendations = self.content_model.predict(
                            book_id, num_recommendations * 2
                        )
                    elif preferences is not None:
                        content_recommendations = self.content_model.predict_by_preferences(
                            preferences, num_recommendations * 2
                        )
                    else:
                        # Get popular books as fallback
                        content_recommendations = self._get_popular_content_books(num_recommendations * 2)

                    logger.info(f"Got {len(content_recommendations)} content recommendations")
                except Exception as e:
                    logger.warning(f"Content recommendations failed: {e}")

            # Combine recommendations
            hybrid_recommendations = self._combine_recommendations(
                collaborative_recommendations,
                content_recommendations,
                num_recommendations
            )

            logger.info(f"Generated {len(hybrid_recommendations)} hybrid recommendations")
            return hybrid_recommendations

        except Exception as e:
            logger.error(f"Failed to generate hybrid recommendations: {e}")
            raise

    def _combine_recommendations(
        self, 
        collab_recs: List[Dict[str, any]], 
        content_recs: List[Dict[str, any]], 
        num_recommendations: int
    ) -> List[Dict[str, any]]:
        """Combine collaborative and content-based recommendations."""

        # Create dictionaries for easy lookup
        collab_dict = {}
        content_dict = {}

        # Process collaborative recommendations
        for rec in collab_recs:
            title = rec.get('title', '')
            score = rec.get('predicted_rating', rec.get('popularity_score', 0))
            collab_dict[title] = {
                'score': score,
                'rank': rec.get('rank', 999),
                'data': rec
            }

        # Process content recommendations
        for rec in content_recs:
            title = rec.get('title', '')
            score = rec.get('similarity_score', 0)
            content_dict[title] = {
                'score': score,
                'rank': rec.get('rank', 999),
                'data': rec
            }

        # Get all unique titles
        all_titles = set(collab_dict.keys()) | set(content_dict.keys())

        # Calculate hybrid scores
        hybrid_scores = []
        for title in all_titles:
            collab_score = collab_dict.get(title, {}).get('score', 0)
            content_score = content_dict.get(title, {}).get('score', 0)

            # Normalize scores (simple min-max normalization)
            collab_score_norm = self._normalize_score(collab_score, [r['score'] for r in collab_dict.values()])
            content_score_norm = self._normalize_score(content_score, [r['score'] for r in content_dict.values()])

            # Calculate hybrid score
            hybrid_score = (
                self.collaborative_weight * collab_score_norm + 
                self.content_weight * content_score_norm
            )

            # Combine recommendation data
            rec_data = {}
            if title in collab_dict:
                rec_data.update(collab_dict[title]['data'])
            if title in content_dict:
                rec_data.update(content_dict[title]['data'])

            rec_data['hybrid_score'] = hybrid_score
            rec_data['collaborative_score'] = collab_score_norm
            rec_data['content_score'] = content_score_norm

            hybrid_scores.append((title, hybrid_score, rec_data))

        # Sort by hybrid score and take top N
        hybrid_scores.sort(key=lambda x: x[1], reverse=True)

        # Format final recommendations
        recommendations = []
        for i, (title, score, rec_data) in enumerate(hybrid_scores[:num_recommendations]):
            recommendation = {
                'title': title,
                'hybrid_score': float(score),
                'collaborative_score': float(rec_data.get('collaborative_score', 0)),
                'content_score': float(rec_data.get('content_score', 0)),
                'rank': i + 1
            }

            # Add additional fields
            additional_fields = ['authors', 'categories', 'book_id', 'average_rating', 'published_year']
            for field in additional_fields:
                if field in rec_data:
                    recommendation[field] = rec_data[field]

            recommendations.append(recommendation)

        return recommendations

    def _normalize_score(self, score: float, all_scores: List[float]) -> float:
        """Normalize a score using min-max normalization."""
        if not all_scores:
            return 0.0

        min_score = min(all_scores)
        max_score = max(all_scores)

        if max_score == min_score:
            return 0.5  # All scores are the same

        return (score - min_score) / (max_score - min_score)

    def _get_popular_content_books(self, num_books: int) -> List[Dict[str, any]]:
        """Get popular books from content model as fallback."""
        if not self.content_model.is_trained:
            return []

        # Simple popularity based on average rating and number of ratings
        books_with_scores = []

        for idx, row in self.content_model.books_df.iterrows():
            avg_rating = row.get('average_rating', 0)
            ratings_count = row.get('ratings_count', 0)

            # Simple popularity score
            popularity_score = avg_rating * np.log(1 + ratings_count)

            books_with_scores.append((idx, popularity_score, row))

        # Sort by popularity
        books_with_scores.sort(key=lambda x: x[1], reverse=True)

        # Format recommendations
        recommendations = []
        for i, (idx, score, row) in enumerate(books_with_scores[:num_books]):
            recommendation = {
                'book_id': int(idx),
                'title': row.get('title', 'Unknown'),
                'authors': row.get('authors', 'Unknown'),
                'categories': row.get('categories', 'Unknown'),
                'similarity_score': float(score),
                'rank': i + 1
            }
            recommendations.append(recommendation)

        return recommendations

    def predict_similar_books(self, book_title: str, num_recommendations: int = 10) -> List[Dict[str, any]]:
        """Find books similar to a given book using hybrid approach."""
        if not self.is_trained:
            raise ModelNotLoadedError("Hybrid model is not trained")

        try:
            collaborative_recommendations = []
            content_recommendations = []

            # Get collaborative recommendations
            if self.collaborative_model.is_trained:
                try:
                    collaborative_recommendations = self.collaborative_model.predict(
                        book_title, num_recommendations * 2
                    )
                except Exception as e:
                    logger.warning(f"Collaborative book similarity failed: {e}")

            # Get content-based recommendations
            if self.content_model.is_trained:
                try:
                    # Find book in content model
                    book_matches = self.content_model.books_df[
                        self.content_model.books_df['title'].str.contains(book_title, case=False, na=False)
                    ]

                    if not book_matches.empty:
                        book_id = book_matches.index[0]
                        content_recommendations = self.content_model.predict(
                            book_id, num_recommendations * 2
                        )
                except Exception as e:
                    logger.warning(f"Content book similarity failed: {e}")

            # Combine recommendations
            return self._combine_recommendations(
                collaborative_recommendations,
                content_recommendations,
                num_recommendations
            )

        except Exception as e:
            logger.error(f"Failed to find similar books: {e}")
            raise

    def get_model_info(self) -> Dict[str, any]:
        """Get information about the hybrid model."""
        info = {
            "status": "trained" if self.is_trained else "not_trained",
            "model_type": "hybrid",
            "weights": {
                "collaborative": self.collaborative_weight,
                "content": self.content_weight
            }
        }

        if self.is_trained:
            info["collaborative_info"] = self.collaborative_model.get_model_info()
            info["content_info"] = self.content_model.get_model_info()

        return info

    def save_model(self, model_path: Optional[str] = None) -> None:
        """Save both component models."""
        if not self.is_trained:
            raise ModelNotLoadedError("No trained model to save")

        # Save both component models
        self.collaborative_model.save_model(model_path)
        self.content_model.save_model(model_path)

        logger.info("Hybrid model saved")

    def load_model(self, model_path: Optional[str] = None) -> None:
        """Load both component models."""
        try:
            self.collaborative_model.load_model(model_path)
            self.content_model.load_model(model_path)
            self.is_trained = True

            logger.info("Hybrid model loaded")
        except Exception as e:
            logger.error(f"Failed to load hybrid model: {e}")
            raise


# Global hybrid model instance
hybrid_model = HybridModel()
