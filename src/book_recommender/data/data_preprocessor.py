"""Data preprocessing utilities for the Book Recommender System."""

import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from book_recommender.core.config import get_settings
from book_recommender.utils.helpers import normalize_text, safe_divide
from book_recommender.utils.constants import ValidationLimits

logger = logging.getLogger(__name__)
settings = get_settings()


class DataPreprocessor:
    """Handles data preprocessing and cleaning."""

    def __init__(self):
        self.settings = get_settings()
        self.scaler = StandardScaler()
        self.min_max_scaler = MinMaxScaler()
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 2)
        )

    def clean_books_data(self, books_df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess books data."""
        logger.info("Cleaning books data")

        # Create a copy to avoid modifying original data
        df = books_df.copy()

        # Remove duplicates based on ISBN
        initial_count = len(df)
        df = df.drop_duplicates(subset=['isbn'], keep='first')
        logger.info(f"Removed {initial_count - len(df)} duplicate books")

        # Clean and validate ISBN
        df['isbn'] = df['isbn'].astype(str).str.strip()
        df = df[df['isbn'].str.len() >= 10]  # Valid ISBN should be at least 10 characters

        # Clean title and author
        df['title'] = df['title'].fillna('Unknown Title')
        df['title'] = df['title'].apply(lambda x: normalize_text(str(x))[:500])

        df['author'] = df['author'].fillna('Unknown Author')
        df['author'] = df['author'].apply(lambda x: normalize_text(str(x))[:200])

        # Clean publication year
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        df['year'] = df['year'].fillna(df['year'].median())
        df['year'] = df['year'].clip(lower=1000, upper=2024)  # Reasonable year range

        # Clean publisher
        df['publisher'] = df['publisher'].fillna('Unknown Publisher')
        df['publisher'] = df['publisher'].apply(lambda x: str(x)[:200])

        # Clean image URL
        if 'image_url' in df.columns:
            df['image_url'] = df['image_url'].fillna('')
            df['image_url'] = df['image_url'].apply(lambda x: str(x) if str(x).startswith('http') else '')

        logger.info(f"Cleaned books data: {len(df)} records")
        return df

    def clean_ratings_data(self, ratings_df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess ratings data."""
        logger.info("Cleaning ratings data")

        df = ratings_df.copy()

        # Remove invalid ratings
        initial_count = len(df)
        df = df[(df['rating'] >= ValidationLimits.MIN_RATING) & 
                (df['rating'] <= ValidationLimits.MAX_RATING)]
        logger.info(f"Removed {initial_count - len(df)} invalid ratings")

        # Remove duplicates (same user rating same book multiple times)
        df = df.drop_duplicates(subset=['user_id', 'isbn'], keep='last')

        # Convert user_id to string for consistency
        df['user_id'] = df['user_id'].astype(str)
        df['isbn'] = df['isbn'].astype(str)

        logger.info(f"Cleaned ratings data: {len(df)} records")
        return df

    def filter_collaborative_data(
        self, 
        ratings_df: pd.DataFrame,
        min_user_ratings: Optional[int] = None,
        min_book_ratings: Optional[int] = None
    ) -> pd.DataFrame:
        """Filter data for collaborative filtering."""
        if min_user_ratings is None:
            min_user_ratings = self.settings.models.collaborative_filtering.min_user_ratings
        if min_book_ratings is None:
            min_book_ratings = self.settings.models.collaborative_filtering.min_book_ratings

        logger.info(f"Filtering collaborative data: min_user_ratings={min_user_ratings}, min_book_ratings={min_book_ratings}")

        df = ratings_df.copy()
        initial_count = len(df)

        # Count ratings per user and book
        user_rating_counts = df['user_id'].value_counts()
        book_rating_counts = df['isbn'].value_counts()

        # Filter users with sufficient ratings
        active_users = user_rating_counts[user_rating_counts >= min_user_ratings].index
        df = df[df['user_id'].isin(active_users)]

        # Filter books with sufficient ratings
        popular_books = book_rating_counts[book_rating_counts >= min_book_ratings].index
        df = df[df['isbn'].isin(popular_books)]

        logger.info(f"Filtered collaborative data: {len(df)} records (removed {initial_count - len(df)})")
        return df

    def create_user_item_matrix(self, ratings_df: pd.DataFrame) -> pd.DataFrame:
        """Create user-item matrix for collaborative filtering."""
        logger.info("Creating user-item matrix")

        # Create pivot table
        user_item_matrix = ratings_df.pivot_table(
            index='user_id',
            columns='isbn',
            values='rating',
            fill_value=0
        )

        logger.info(f"Created user-item matrix: {user_item_matrix.shape}")
        return user_item_matrix

    def create_book_pivot(self, books_df: pd.DataFrame, ratings_df: pd.DataFrame) -> pd.DataFrame:
        """Create book pivot table for recommendations."""
        logger.info("Creating book pivot table")

        # Merge books with ratings
        merged_df = ratings_df.merge(books_df[['isbn', 'title']], on='isbn', how='inner')

        # Create pivot table with books as index
        book_pivot = merged_df.pivot_table(
            index='title',
            columns='user_id',
            values='rating',
            fill_value=0
        )

        logger.info(f"Created book pivot table: {book_pivot.shape}")
        return book_pivot

    def prepare_content_features(self, books_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for content-based filtering."""
        logger.info("Preparing content features")

        df = books_df.copy()

        # Create textual representation
        df['textual_representation'] = df.apply(self._create_textual_representation, axis=1)

        # Normalize numerical features if available
        numerical_features = ['average_rating', 'num_pages', 'ratings_count']
        available_features = [col for col in numerical_features if col in df.columns]

        for feature in available_features:
            df[feature] = pd.to_numeric(df[feature], errors='coerce')
            df[feature] = df[feature].fillna(df[feature].median())
            df[f'{feature}_normalized'] = self.min_max_scaler.fit_transform(df[[feature]])

        # Extract year decade for categorical grouping
        if 'published_year' in df.columns:
            df['published_year'] = pd.to_numeric(df['published_year'], errors='coerce')
            df['decade'] = (df['published_year'] // 10) * 10

        logger.info(f"Prepared content features for {len(df)} books")
        return df

    def _create_textual_representation(self, row: pd.Series) -> str:
        """Create textual representation for a book."""
        components = []

        # Add title
        if pd.notna(row.get('title')):
            components.append(f"Title: {row['title']}")

        # Add authors
        if pd.notna(row.get('authors')):
            components.append(f"Authors: {row['authors']}")

        # Add description
        if pd.notna(row.get('description')):
            # Limit description length
            desc = str(row['description'])[:1000]
            components.append(f"Description: {desc}")

        # Add categories
        if pd.notna(row.get('categories')):
            components.append(f"Categories: {row['categories']}")

        # Add other metadata
        for col in ['published_year', 'average_rating', 'num_pages']:
            if col in row and pd.notna(row[col]):
                components.append(f"{col.replace('_', ' ').title()}: {row[col]}")

        return "\n ".join(components)

    def extract_tfidf_features(self, texts: List[str]) -> np.ndarray:
        """Extract TF-IDF features from text data."""
        logger.info(f"Extracting TF-IDF features from {len(texts)} texts")

        # Clean and normalize texts
        cleaned_texts = [normalize_text(text) for text in texts]

        # Fit and transform
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(cleaned_texts)

        logger.info(f"Extracted TF-IDF features: {tfidf_matrix.shape}")
        return tfidf_matrix.toarray()

    def calculate_statistics(self, df: pd.DataFrame) -> Dict[str, any]:
        """Calculate dataset statistics."""
        stats = {}

        # Basic statistics
        stats['total_records'] = len(df)
        stats['memory_usage'] = df.memory_usage(deep=True).sum()

        # Missing values
        stats['missing_values'] = df.isnull().sum().to_dict()
        stats['missing_percentage'] = (df.isnull().sum() / len(df) * 100).to_dict()

        # Duplicates
        stats['duplicate_records'] = df.duplicated().sum()

        # Data types
        stats['data_types'] = df.dtypes.astype(str).to_dict()

        # For numerical columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_columns:
            stats['numeric_summary'] = df[numeric_columns].describe().to_dict()

        # For categorical columns
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        if categorical_columns:
            stats['categorical_summary'] = {}
            for col in categorical_columns:
                stats['categorical_summary'][col] = {
                    'unique_count': df[col].nunique(),
                    'most_frequent': df[col].mode().iloc[0] if not df[col].mode().empty else None,
                    'frequency': df[col].value_counts().head().to_dict()
                }

        return stats

    def split_data(
        self, 
        df: pd.DataFrame, 
        train_size: float = 0.8, 
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into train and test sets."""
        logger.info(f"Splitting data: train_size={train_size}")

        # Shuffle data
        df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

        # Split
        split_index = int(len(df_shuffled) * train_size)
        train_df = df_shuffled[:split_index]
        test_df = df_shuffled[split_index:]

        logger.info(f"Train set: {len(train_df)} records, Test set: {len(test_df)} records")
        return train_df, test_df


# Global preprocessor instance
data_preprocessor = DataPreprocessor()
