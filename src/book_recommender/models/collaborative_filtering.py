"""Collaborative Filtering model for the Book Recommender System."""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

from book_recommender.core.config import get_settings
from book_recommender.core.exceptions import ModelNotLoadedError, DataNotFoundError
from book_recommender.data.data_loader import data_loader
from book_recommender.data.data_preprocessor import data_preprocessor
from book_recommender.utils.helpers import ensure_directory

logger = logging.getLogger(__name__)
settings = get_settings()


class CollaborativeFilteringModel:
    """Collaborative Filtering recommendation model using K-Nearest Neighbors."""

    def __init__(self):
        self.settings = get_settings()
        self.model = None
        self.book_pivot = None
        self.book_names = None
        self.user_item_matrix = None
        self.similarity_matrix = None
        self.is_trained = False

        # Model parameters
        self.n_neighbors = self.settings.models.collaborative_filtering.n_neighbors
        self.algorithm = self.settings.models.collaborative_filtering.algorithm
        self.min_user_ratings = self.settings.models.collaborative_filtering.min_user_ratings
        self.min_book_ratings = self.settings.models.collaborative_filtering.min_book_ratings

        # Ensure model directory exists
        self.models_path = Path(self.settings.models_path)
        ensure_directory(self.models_path)

    def train(self, ratings_df: pd.DataFrame, books_df: pd.DataFrame) -> Dict[str, any]:
        """Train the collaborative filtering model."""
        logger.info("Training Collaborative Filtering model")

        try:
            # Preprocess data
            logger.info("Preprocessing data for collaborative filtering")
            filtered_ratings = data_preprocessor.filter_collaborative_data(
                ratings_df, 
                self.min_user_ratings, 
                self.min_book_ratings
            )

            # Create book pivot table
            self.book_pivot = data_preprocessor.create_book_pivot(books_df, filtered_ratings)
            logger.info(f"Created book pivot table with shape: {self.book_pivot.shape}")

            # Create sparse matrix for efficient computation
            book_sparse = csr_matrix(self.book_pivot.values)

            # Initialize and train KNN model
            self.model = NearestNeighbors(
                n_neighbors=self.n_neighbors,
                algorithm=self.algorithm,
                metric='cosine'
            )

            self.model.fit(book_sparse)
            logger.info(f"Trained KNN model with {self.n_neighbors} neighbors")

            # Store book names for easy lookup
            self.book_names = self.book_pivot.index.tolist()

            # Calculate similarity matrix for analysis
            self.similarity_matrix = cosine_similarity(book_sparse)
            logger.info("Calculated similarity matrix")

            self.is_trained = True

            # Training statistics
            training_stats = {
                'total_books': len(self.book_names),
                'total_users': len(self.book_pivot.columns),
                'sparsity': 1 - (np.count_nonzero(self.book_pivot.values) / self.book_pivot.size),
                'avg_ratings_per_book': self.book_pivot.sum(axis=1).mean(),
                'avg_ratings_per_user': self.book_pivot.sum(axis=0).mean()
            }

            logger.info(f"Collaborative filtering training completed: {training_stats}")
            return training_stats

        except Exception as e:
            logger.error(f"Failed to train collaborative filtering model: {e}")
            raise

    def predict(self, book_title: str, num_recommendations: int = 5) -> List[Dict[str, any]]:
        """Generate recommendations based on a book title."""
        if not self.is_trained:
            raise ModelNotLoadedError("Collaborative filtering model is not trained")

        try:
            # Find book index
            if book_title not in self.book_names:
                raise DataNotFoundError(f"Book '{book_title}' not found in training data")

            book_index = self.book_names.index(book_title)
            logger.info(f"Generating recommendations for book: {book_title} (index: {book_index})")

            # Get similar books
            book_vector = self.book_pivot.iloc[book_index, :].values.reshape(1, -1)
            distances, suggestions = self.model.kneighbors(book_vector, n_neighbors=num_recommendations + 1)

            # Format recommendations (skip the first one as it's the input book itself)
            recommendations = []
            for i in range(1, len(suggestions[0])):
                book_idx = suggestions[0][i]
                similarity_score = 1 - distances[0][i]  # Convert distance to similarity

                recommendation = {
                    'title': self.book_names[book_idx],
                    'similarity_score': float(similarity_score),
                    'rank': i
                }
                recommendations.append(recommendation)

            logger.info(f"Generated {len(recommendations)} recommendations")
            return recommendations

        except Exception as e:
            logger.error(f"Failed to generate recommendations: {e}")
            raise

    def predict_for_user(self, user_id: str, num_recommendations: int = 10) -> List[Dict[str, any]]:
        """Generate recommendations for a specific user."""
        if not self.is_trained:
            raise ModelNotLoadedError("Collaborative filtering model is not trained")

        try:
            # Check if user exists in the pivot table
            if user_id not in self.book_pivot.columns:
                # For new users, return popular books
                logger.info(f"User {user_id} not found, returning popular books")
                return self._get_popular_books(num_recommendations)

            # Get user's ratings
            user_ratings = self.book_pivot[user_id]
            rated_books = user_ratings[user_ratings > 0].index.tolist()

            if len(rated_books) == 0:
                logger.info(f"User {user_id} has no ratings, returning popular books")
                return self._get_popular_books(num_recommendations)

            # Find similar users
            user_vector = user_ratings.values.reshape(1, -1)
            distances, similar_users_indices = self.model.kneighbors(user_vector, n_neighbors=min(10, len(self.book_pivot.columns)))

            # Get recommendations based on similar users
            recommendations_scores = {}

            for user_idx in similar_users_indices[0][1:]:  # Skip the user itself
                similar_user_id = self.book_pivot.columns[user_idx]
                similar_user_ratings = self.book_pivot[similar_user_id]

                # Get books rated highly by similar user but not rated by target user
                highly_rated_books = similar_user_ratings[similar_user_ratings >= 4].index.tolist()

                for book in highly_rated_books:
                    if book not in rated_books:
                        if book not in recommendations_scores:
                            recommendations_scores[book] = 0
                        recommendations_scores[book] += similar_user_ratings[book]

            # Sort recommendations by score
            sorted_recommendations = sorted(
                recommendations_scores.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:num_recommendations]

            # Format recommendations
            recommendations = []
            for rank, (book_title, score) in enumerate(sorted_recommendations, 1):
                recommendation = {
                    'title': book_title,
                    'predicted_rating': float(score / len(similar_users_indices[0][1:])),
                    'rank': rank
                }
                recommendations.append(recommendation)

            logger.info(f"Generated {len(recommendations)} user-based recommendations")
            return recommendations

        except Exception as e:
            logger.error(f"Failed to generate user recommendations: {e}")
            raise

    def _get_popular_books(self, num_recommendations: int) -> List[Dict[str, any]]:
        """Get popular books as fallback recommendations."""
        if not self.is_trained:
            return []

        # Calculate book popularity (number of ratings and average rating)
        book_stats = []
        for book_title in self.book_names:
            book_ratings = self.book_pivot.loc[book_title]
            num_ratings = (book_ratings > 0).sum()
            avg_rating = book_ratings[book_ratings > 0].mean() if num_ratings > 0 else 0

            popularity_score = num_ratings * avg_rating  # Simple popularity metric
            book_stats.append((book_title, popularity_score, num_ratings, avg_rating))

        # Sort by popularity score
        book_stats.sort(key=lambda x: x[1], reverse=True)

        # Format recommendations
        recommendations = []
        for rank, (title, popularity_score, num_ratings, avg_rating) in enumerate(book_stats[:num_recommendations], 1):
            recommendation = {
                'title': title,
                'popularity_score': float(popularity_score),
                'num_ratings': int(num_ratings),
                'avg_rating': float(avg_rating),
                'rank': rank
            }
            recommendations.append(recommendation)

        return recommendations

    def get_book_similarity(self, book1: str, book2: str) -> float:
        """Get similarity score between two books."""
        if not self.is_trained:
            raise ModelNotLoadedError("Collaborative filtering model is not trained")

        try:
            if book1 not in self.book_names or book2 not in self.book_names:
                raise DataNotFoundError("One or both books not found in training data")

            book1_idx = self.book_names.index(book1)
            book2_idx = self.book_names.index(book2)

            similarity_score = self.similarity_matrix[book1_idx][book2_idx]
            return float(similarity_score)

        except Exception as e:
            logger.error(f"Failed to calculate book similarity: {e}")
            raise

    def save_model(self, model_path: Optional[Path] = None) -> None:
        """Save the trained model to disk."""
        if not self.is_trained:
            raise ModelNotLoadedError("No trained model to save")

        if model_path is None:
            model_path = self.models_path

        try:
            ensure_directory(model_path)

            # Save KNN model
            with open(model_path / "collaborative_knn_model.pkl", "wb") as f:
                pickle.dump(self.model, f)

            # Save book pivot
            with open(model_path / "book_pivot.pkl", "wb") as f:
                pickle.dump(self.book_pivot, f)

            # Save book names
            with open(model_path / "book_names.pkl", "wb") as f:
                pickle.dump(self.book_names, f)

            # Save similarity matrix
            with open(model_path / "similarity_matrix.pkl", "wb") as f:
                pickle.dump(self.similarity_matrix, f)

            logger.info(f"Collaborative filtering model saved to {model_path}")

        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise

    def load_model(self, model_path: Optional[Path] = None) -> None:
        """Load a trained model from disk."""
        if model_path is None:
            model_path = self.models_path

        try:
            # Load KNN model
            with open(model_path / "collaborative_knn_model.pkl", "rb") as f:
                self.model = pickle.load(f)

            # Load book pivot
            with open(model_path / "book_pivot.pkl", "rb") as f:
                self.book_pivot = pickle.load(f)

            # Load book names
            with open(model_path / "book_names.pkl", "rb") as f:
                self.book_names = pickle.load(f)

            # Load similarity matrix
            with open(model_path / "similarity_matrix.pkl", "rb") as f:
                self.similarity_matrix = pickle.load(f)

            self.is_trained = True
            logger.info(f"Collaborative filtering model loaded from {model_path}")

        except FileNotFoundError as e:
            logger.error(f"Model files not found: {e}")
            raise ModelNotLoadedError(f"Could not load model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def get_model_info(self) -> Dict[str, any]:
        """Get information about the trained model."""
        if not self.is_trained:
            return {"status": "not_trained"}

        info = {
            "status": "trained",
            "model_type": "collaborative_filtering",
            "algorithm": self.algorithm,
            "n_neighbors": self.n_neighbors,
            "total_books": len(self.book_names),
            "total_users": len(self.book_pivot.columns),
            "sparsity": float(1 - (np.count_nonzero(self.book_pivot.values) / self.book_pivot.size)),
            "parameters": {
                "min_user_ratings": self.min_user_ratings,
                "min_book_ratings": self.min_book_ratings
            }
        }

        return info


# Global collaborative filtering model instance
collaborative_model = CollaborativeFilteringModel()
