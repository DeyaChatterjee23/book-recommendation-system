"""Data loading utilities for the Book Recommender System."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError

from book_recommender.core.config import get_settings
from book_recommender.core.exceptions import DataNotFoundError, ValidationError
from book_recommender.utils.helpers import ensure_directory

logger = logging.getLogger(__name__)
settings = get_settings()


class DataLoader:
    """Handles loading data from various sources."""

    def __init__(self):
        self.settings = get_settings()
        self.raw_data_path = Path(self.settings.raw_data_path)
        self.processed_data_path = Path(self.settings.processed_data_path)

        # Ensure directories exist
        ensure_directory(self.raw_data_path)
        ensure_directory(self.processed_data_path)

    def load_books_data(self, file_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
        """Load books dataset."""
        if file_path is None:
            file_path = self.raw_data_path / "BX-Books.csv"

        try:
            logger.info(f"Loading books data from {file_path}")

            # Try different encodings and separators
            encodings = ['latin-1', 'utf-8', 'iso-8859-1', 'cp1252']
            separators = [';', ',', '\t']

            books_df = None
            for encoding in encodings:
                for sep in separators:
                    try:
                        books_df = pd.read_csv(
                            file_path,
                            sep=sep,
                            encoding=encoding,
                            on_bad_lines='skip'
                        )
                        if len(books_df.columns) >= 6:  # Expected number of columns
                            logger.info(f"Successfully loaded books data with encoding {encoding} and separator '{sep}'")
                            break
                    except Exception:
                        continue
                if books_df is not None and len(books_df.columns) >= 6:
                    break

            if books_df is None:
                raise DataNotFoundError(f"Could not load books data from {file_path}")

            # Standardize column names
            books_df.columns = books_df.columns.str.lower().str.replace('-', '_')
            expected_columns = ['isbn', 'book_title', 'book_author', 'year_of_publication', 'publisher']

            # Rename columns to standard format
            column_mapping = {
                'book_title': 'title',
                'book_author': 'author',
                'year_of_publication': 'year',
                'image_url_l': 'image_url'
            }

            books_df = books_df.rename(columns=column_mapping)

            logger.info(f"Loaded {len(books_df)} books")
            return books_df

        except Exception as e:
            logger.error(f"Failed to load books data: {e}")
            raise DataNotFoundError(f"Could not load books data: {e}")

    def load_users_data(self, file_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
        """Load users dataset."""
        if file_path is None:
            file_path = self.raw_data_path / "BX-Users.csv"

        try:
            logger.info(f"Loading users data from {file_path}")

            users_df = pd.read_csv(
                file_path,
                sep=';',
                encoding='latin-1',
                on_bad_lines='skip'
            )

            # Standardize column names
            users_df.columns = users_df.columns.str.lower().str.replace('-', '_')

            logger.info(f"Loaded {len(users_df)} users")
            return users_df

        except Exception as e:
            logger.error(f"Failed to load users data: {e}")
            raise DataNotFoundError(f"Could not load users data: {e}")

    def load_ratings_data(self, file_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
        """Load ratings dataset."""
        if file_path is None:
            file_path = self.raw_data_path / "BX-Book-Ratings.csv"

        try:
            logger.info(f"Loading ratings data from {file_path}")

            ratings_df = pd.read_csv(
                file_path,
                sep=';',
                encoding='latin-1',
                on_bad_lines='skip'
            )

            # Standardize column names
            ratings_df.columns = ratings_df.columns.str.lower().str.replace('-', '_')
            column_mapping = {
                'user_id': 'user_id',
                'book_rating': 'rating'
            }
            ratings_df = ratings_df.rename(columns=column_mapping)

            logger.info(f"Loaded {len(ratings_df)} ratings")
            return ratings_df

        except Exception as e:
            logger.error(f"Failed to load ratings data: {e}")
            raise DataNotFoundError(f"Could not load ratings data: {e}")

    def load_content_books_data(self, file_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
        """Load content-based books dataset (7kbooks.csv)."""
        if file_path is None:
            file_path = self.raw_data_path / "7kbooks.csv"

        try:
            logger.info(f"Loading content books data from {file_path}")

            books_df = pd.read_csv(file_path, encoding='utf-8')

            # Validate required columns
            required_columns = ['title', 'authors', 'description', 'categories']
            missing_columns = [col for col in required_columns if col not in books_df.columns]

            if missing_columns:
                raise ValidationError(f"Missing required columns: {missing_columns}")

            logger.info(f"Loaded {len(books_df)} content books")
            return books_df

        except Exception as e:
            logger.error(f"Failed to load content books data: {e}")
            raise DataNotFoundError(f"Could not load content books data: {e}")

    def load_processed_data(self, filename: str) -> pd.DataFrame:
        """Load processed data file."""
        file_path = self.processed_data_path / filename

        if not file_path.exists():
            raise DataNotFoundError(f"Processed data file not found: {file_path}")

        try:
            if filename.endswith('.csv'):
                return pd.read_csv(file_path)
            elif filename.endswith('.parquet'):
                return pd.read_parquet(file_path)
            elif filename.endswith('.pkl'):
                return pd.read_pickle(file_path)
            else:
                raise ValueError(f"Unsupported file format: {filename}")
        except Exception as e:
            logger.error(f"Failed to load processed data {filename}: {e}")
            raise DataNotFoundError(f"Could not load processed data: {e}")

    def save_processed_data(self, data: pd.DataFrame, filename: str) -> None:
        """Save processed data to file."""
        file_path = self.processed_data_path / filename

        try:
            if filename.endswith('.csv'):
                data.to_csv(file_path, index=False)
            elif filename.endswith('.parquet'):
                data.to_parquet(file_path, index=False)
            elif filename.endswith('.pkl'):
                data.to_pickle(file_path)
            else:
                raise ValueError(f"Unsupported file format: {filename}")

            logger.info(f"Saved processed data to {file_path}")

        except Exception as e:
            logger.error(f"Failed to save processed data {filename}: {e}")
            raise

    def load_from_database(self, query: str) -> pd.DataFrame:
        """Load data from database using SQL query."""
        try:
            engine = create_engine(self.settings.database_url)
            df = pd.read_sql_query(query, engine)
            logger.info(f"Loaded {len(df)} rows from database")
            return df
        except SQLAlchemyError as e:
            logger.error(f"Database query failed: {e}")
            raise DataNotFoundError(f"Could not load data from database: {e}")

    def get_data_info(self, df: pd.DataFrame) -> Dict[str, any]:
        """Get comprehensive information about a dataset."""
        info = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'null_counts': df.isnull().sum().to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'duplicate_rows': df.duplicated().sum()
        }

        # Add numerical column statistics
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_columns:
            info['numeric_stats'] = df[numeric_columns].describe().to_dict()

        # Add categorical column statistics
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        if categorical_columns:
            info['categorical_stats'] = {}
            for col in categorical_columns:
                info['categorical_stats'][col] = {
                    'unique_values': df[col].nunique(),
                    'most_common': df[col].mode().iloc[0] if not df[col].mode().empty else None
                }

        return info


# Global data loader instance
data_loader = DataLoader()
