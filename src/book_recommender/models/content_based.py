"""Content-Based Filtering model for the Book Recommender System."""

import logging
import pickle
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import faiss
import aiohttp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from book_recommender.core.config import get_settings
from book_recommender.core.exceptions import ModelNotLoadedError, DataNotFoundError, ExternalServiceError
from book_recommender.data.data_loader import data_loader
from book_recommender.data.data_preprocessor import data_preprocessor
from book_recommender.utils.helpers import ensure_directory, retry_async

logger = logging.getLogger(__name__)
settings = get_settings()


class ContentBasedModel:
    """Content-Based Filtering recommendation model using text embeddings and FAISS."""

    def __init__(self):
        self.settings = get_settings()
        self.books_df = None
        self.embeddings = None
        self.faiss_index = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.is_trained = False

        # Model parameters
        self.embedding_dim = self.settings.models.content_based.embedding_dim
        self.ollama_url = self.settings.models.content_based.ollama_url
        self.ollama_model = self.settings.models.content_based.ollama_model

        # Ensure model directory exists
        self.models_path = Path(self.settings.models_path)
        ensure_directory(self.models_path)

        # Initialize TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )

    async def train(self, books_df: pd.DataFrame, use_ollama: bool = True) -> Dict[str, any]:
        """Train the content-based filtering model."""
        logger.info("Training Content-Based Filtering model")

        try:
            # Prepare content features
            self.books_df = data_preprocessor.prepare_content_features(books_df)
            logger.info(f"Prepared content features for {len(self.books_df)} books")

            if use_ollama:
                # Use Ollama for embeddings
                await self._create_ollama_embeddings()
            else:
                # Use TF-IDF as fallback
                self._create_tfidf_embeddings()

            self.is_trained = True

            # Training statistics
            training_stats = {
                'total_books': len(self.books_df),
                'embedding_method': 'ollama' if use_ollama else 'tfidf',
                'embedding_dimension': self.embedding_dim if use_ollama else self.tfidf_matrix.shape[1],
                'avg_text_length': self.books_df['textual_representation'].str.len().mean()
            }

            logger.info(f"Content-based filtering training completed: {training_stats}")
            return training_stats

        except Exception as e:
            logger.error(f"Failed to train content-based model: {e}")
            raise

    async def _create_ollama_embeddings(self):
        """Create embeddings using Ollama API."""
        logger.info("Creating embeddings using Ollama")

        # Initialize FAISS index
        self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)
        embeddings_list = []

        async with aiohttp.ClientSession() as session:
            for idx, row in self.books_df.iterrows():
                if idx % 100 == 0:
                    logger.info(f"Processed {idx}/{len(self.books_df)} books")

                try:
                    embedding = await self._get_ollama_embedding(
                        session, 
                        row['textual_representation']
                    )
                    embeddings_list.append(embedding)

                except Exception as e:
                    logger.warning(f"Failed to get embedding for book {idx}: {e}")
                    # Use zero embedding as fallback
                    embeddings_list.append(np.zeros(self.embedding_dim))

        # Convert to numpy array and add to FAISS index
        self.embeddings = np.array(embeddings_list, dtype=np.float32)
        self.faiss_index.add(self.embeddings)

        logger.info(f"Created {len(embeddings_list)} embeddings using Ollama")

    async def _get_ollama_embedding(self, session: aiohttp.ClientSession, text: str) -> np.ndarray:
        """Get embedding from Ollama API with retry logic."""

        async def make_request():
            async with session.post(
                f"{self.ollama_url}/api/embeddings",
                json={
                    'model': self.ollama_model,
                    'prompt': text
                },
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status != 200:
                    raise ExternalServiceError(f"Ollama API returned status {response.status}")

                data = await response.json()
                if 'embedding' not in data:
                    raise ExternalServiceError("No embedding in Ollama response")

                return np.array(data['embedding'], dtype=np.float32)

        return await retry_async(make_request, max_retries=3, delay=1.0)

    def _create_tfidf_embeddings(self):
        """Create TF-IDF embeddings as fallback."""
        logger.info("Creating TF-IDF embeddings")

        # Extract text data
        texts = self.books_df['textual_representation'].fillna('').tolist()

        # Create TF-IDF matrix
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)

        # Create FAISS index for TF-IDF (using inner product for cosine similarity)
        self.faiss_index = faiss.IndexFlatIP(self.tfidf_matrix.shape[1])

        # Normalize TF-IDF vectors for cosine similarity
        tfidf_normalized = self.tfidf_matrix.copy()
        faiss.normalize_L2(tfidf_normalized.toarray().astype(np.float32))

        self.faiss_index.add(tfidf_normalized.toarray().astype(np.float32))

        logger.info(f"Created TF-IDF embeddings with {self.tfidf_matrix.shape[1]} features")

    def predict(self, book_id: int, num_recommendations: int = 10) -> List[Dict[str, any]]:
        """Generate content-based recommendations for a book."""
        if not self.is_trained:
            raise ModelNotLoadedError("Content-based model is not trained")

        try:
            if book_id >= len(self.books_df) or book_id < 0:
                raise DataNotFoundError(f"Book ID {book_id} not found")

            logger.info(f"Generating content-based recommendations for book ID: {book_id}")

            # Get book information
            book_info = self.books_df.iloc[book_id]

            # Search for similar books using FAISS
            if self.embeddings is not None:
                # Using Ollama embeddings
                query_embedding = self.embeddings[book_id].reshape(1, -1)
            else:
                # Using TF-IDF embeddings
                query_embedding = self.tfidf_matrix[book_id].toarray().astype(np.float32)
                faiss.normalize_L2(query_embedding)

            # Search for similar books
            distances, similar_indices = self.faiss_index.search(
                query_embedding, 
                num_recommendations + 1  # +1 to exclude the query book itself
            )

            # Format recommendations
            recommendations = []
            for i, (idx, distance) in enumerate(zip(similar_indices[0], distances[0])):
                if idx == book_id:  # Skip the query book itself
                    continue

                similar_book = self.books_df.iloc[idx]

                # Calculate similarity score (convert distance to similarity)
                if self.embeddings is not None:
                    similarity_score = 1 / (1 + distance)  # For L2 distance
                else:
                    similarity_score = distance  # For inner product (cosine similarity)

                recommendation = {
                    'book_id': int(idx),
                    'title': similar_book.get('title', 'Unknown'),
                    'authors': similar_book.get('authors', 'Unknown'),
                    'categories': similar_book.get('categories', 'Unknown'),
                    'similarity_score': float(similarity_score),
                    'rank': len(recommendations) + 1
                }

                # Add additional metadata if available
                if 'average_rating' in similar_book:
                    recommendation['average_rating'] = float(similar_book['average_rating'])
                if 'published_year' in similar_book:
                    recommendation['published_year'] = similar_book['published_year']

                recommendations.append(recommendation)

                if len(recommendations) >= num_recommendations:
                    break

            logger.info(f"Generated {len(recommendations)} content-based recommendations")
            return recommendations

        except Exception as e:
            logger.error(f"Failed to generate content-based recommendations: {e}")
            raise

    def predict_by_preferences(self, preferences: Dict[str, any], num_recommendations: int = 10) -> List[Dict[str, any]]:
        """Generate recommendations based on user preferences."""
        if not self.is_trained:
            raise ModelNotLoadedError("Content-based model is not trained")

        try:
            logger.info(f"Generating recommendations based on preferences: {preferences}")

            # Create preference text
            preference_text = self._create_preference_text(preferences)

            # Get embedding for preferences
            if self.embeddings is not None:
                # For Ollama embeddings, we'd need to get embedding via API
                # For now, use similarity to existing books
                recommendations = self._find_books_by_text_similarity(preference_text, num_recommendations)
            else:
                # Use TF-IDF similarity
                preference_vector = self.tfidf_vectorizer.transform([preference_text])
                preference_normalized = preference_vector.copy()
                faiss.normalize_L2(preference_normalized.toarray().astype(np.float32))

                # Search for similar books
                distances, similar_indices = self.faiss_index.search(
                    preference_normalized.toarray().astype(np.float32),
                    num_recommendations
                )

                recommendations = []
                for i, (idx, distance) in enumerate(zip(similar_indices[0], distances[0])):
                    book = self.books_df.iloc[idx]

                    recommendation = {
                        'book_id': int(idx),
                        'title': book.get('title', 'Unknown'),
                        'authors': book.get('authors', 'Unknown'),
                        'categories': book.get('categories', 'Unknown'),
                        'similarity_score': float(distance),
                        'rank': i + 1
                    }
                    recommendations.append(recommendation)

            logger.info(f"Generated {len(recommendations)} preference-based recommendations")
            return recommendations

        except Exception as e:
            logger.error(f"Failed to generate preference-based recommendations: {e}")
            raise

    def _create_preference_text(self, preferences: Dict[str, any]) -> str:
        """Create text representation from user preferences."""
        text_components = []

        if 'genres' in preferences:
            text_components.append(f"Categories: {', '.join(preferences['genres'])}")

        if 'authors' in preferences:
            text_components.append(f"Authors: {', '.join(preferences['authors'])}")

        if 'keywords' in preferences:
            text_components.append(f"Keywords: {', '.join(preferences['keywords'])}")

        if 'description' in preferences:
            text_components.append(f"Description: {preferences['description']}")

        return "\n ".join(text_components)

    def _find_books_by_text_similarity(self, query_text: str, num_recommendations: int) -> List[Dict[str, any]]:
        """Find books similar to query text using TF-IDF similarity."""
        # This is a fallback method when Ollama embeddings are not available
        if self.tfidf_matrix is None:
            return []

        query_vector = self.tfidf_vectorizer.transform([query_text])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

        # Get top similar books
        similar_indices = np.argsort(similarities)[::-1][:num_recommendations]

        recommendations = []
        for i, idx in enumerate(similar_indices):
            book = self.books_df.iloc[idx]

            recommendation = {
                'book_id': int(idx),
                'title': book.get('title', 'Unknown'),
                'authors': book.get('authors', 'Unknown'),
                'categories': book.get('categories', 'Unknown'),
                'similarity_score': float(similarities[idx]),
                'rank': i + 1
            }
            recommendations.append(recommendation)

        return recommendations

    def get_book_details(self, book_id: int) -> Dict[str, any]:
        """Get detailed information about a book."""
        if not self.is_trained:
            raise ModelNotLoadedError("Content-based model is not trained")

        if book_id >= len(self.books_df) or book_id < 0:
            raise DataNotFoundError(f"Book ID {book_id} not found")

        book = self.books_df.iloc[book_id]

        details = {
            'book_id': book_id,
            'title': book.get('title', 'Unknown'),
            'authors': book.get('authors', 'Unknown'),
            'categories': book.get('categories', 'Unknown'),
            'description': book.get('description', ''),
        }

        # Add optional fields
        optional_fields = ['published_year', 'average_rating', 'num_pages', 'ratings_count', 'isbn13', 'isbn10']
        for field in optional_fields:
            if field in book and pd.notna(book[field]):
                details[field] = book[field]

        return details

    def search_books(self, query: str, num_results: int = 20) -> List[Dict[str, any]]:
        """Search books by text query."""
        if not self.is_trained:
            raise ModelNotLoadedError("Content-based model is not trained")

        try:
            # Simple text search in titles and descriptions
            query_lower = query.lower()

            matching_books = []
            for idx, row in self.books_df.iterrows():
                title = str(row.get('title', '')).lower()
                authors = str(row.get('authors', '')).lower()
                description = str(row.get('description', '')).lower()
                categories = str(row.get('categories', '')).lower()

                # Calculate relevance score
                score = 0
                if query_lower in title:
                    score += 3
                if query_lower in authors:
                    score += 2
                if query_lower in categories:
                    score += 2
                if query_lower in description:
                    score += 1

                if score > 0:
                    matching_books.append((idx, score))

            # Sort by relevance score
            matching_books.sort(key=lambda x: x[1], reverse=True)

            # Format results
            results = []
            for idx, score in matching_books[:num_results]:
                book = self.books_df.iloc[idx]
                result = {
                    'book_id': int(idx),
                    'title': book.get('title', 'Unknown'),
                    'authors': book.get('authors', 'Unknown'),
                    'categories': book.get('categories', 'Unknown'),
                    'relevance_score': score
                }
                results.append(result)

            logger.info(f"Found {len(results)} books matching query: {query}")
            return results

        except Exception as e:
            logger.error(f"Failed to search books: {e}")
            raise

    def save_model(self, model_path: Optional[Path] = None) -> None:
        """Save the trained model to disk."""
        if not self.is_trained:
            raise ModelNotLoadedError("No trained model to save")

        if model_path is None:
            model_path = self.models_path

        try:
            ensure_directory(model_path)

            # Save books DataFrame
            self.books_df.to_pickle(model_path / "content_books_df.pkl")

            # Save FAISS index
            if self.faiss_index is not None:
                faiss.write_index(self.faiss_index, str(model_path / "content_faiss_index.faiss"))

            # Save embeddings if using Ollama
            if self.embeddings is not None:
                np.save(model_path / "content_embeddings.npy", self.embeddings)

            # Save TF-IDF vectorizer and matrix if using TF-IDF
            if self.tfidf_vectorizer is not None:
                with open(model_path / "tfidf_vectorizer.pkl", "wb") as f:
                    pickle.dump(self.tfidf_vectorizer, f)

            if self.tfidf_matrix is not None:
                with open(model_path / "tfidf_matrix.pkl", "wb") as f:
                    pickle.dump(self.tfidf_matrix, f)

            logger.info(f"Content-based model saved to {model_path}")

        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise

    def load_model(self, model_path: Optional[Path] = None) -> None:
        """Load a trained model from disk."""
        if model_path is None:
            model_path = self.models_path

        try:
            # Load books DataFrame
            self.books_df = pd.read_pickle(model_path / "content_books_df.pkl")

            # Load FAISS index
            faiss_path = model_path / "content_faiss_index.faiss"
            if faiss_path.exists():
                self.faiss_index = faiss.read_index(str(faiss_path))

            # Load embeddings if available
            embeddings_path = model_path / "content_embeddings.npy"
            if embeddings_path.exists():
                self.embeddings = np.load(embeddings_path)

            # Load TF-IDF components if available
            tfidf_vectorizer_path = model_path / "tfidf_vectorizer.pkl"
            if tfidf_vectorizer_path.exists():
                with open(tfidf_vectorizer_path, "rb") as f:
                    self.tfidf_vectorizer = pickle.load(f)

            tfidf_matrix_path = model_path / "tfidf_matrix.pkl"
            if tfidf_matrix_path.exists():
                with open(tfidf_matrix_path, "rb") as f:
                    self.tfidf_matrix = pickle.load(f)

            self.is_trained = True
            logger.info(f"Content-based model loaded from {model_path}")

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
            "model_type": "content_based",
            "total_books": len(self.books_df),
            "embedding_method": "ollama" if self.embeddings is not None else "tfidf",
            "embedding_dimension": self.embedding_dim if self.embeddings is not None else (self.tfidf_matrix.shape[1] if self.tfidf_matrix is not None else 0),
            "parameters": {
                "ollama_model": self.ollama_model,
                "embedding_dim": self.embedding_dim
            }
        }

        return info


# Global content-based model instance
content_model = ContentBasedModel()
