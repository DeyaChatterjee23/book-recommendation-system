"""Data validation utilities for the Book Recommender System."""

import logging
import re
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import numpy as np
from pydantic import BaseModel, validator, ValidationError as PydanticValidationError

from book_recommender.core.exceptions import ValidationError
from book_recommender.utils.constants import ValidationLimits, RegexPatterns
from book_recommender.utils.helpers import validate_isbn

logger = logging.getLogger(__name__)


class BookModel(BaseModel):
    """Pydantic model for book validation."""
    isbn: str
    title: str
    author: str
    year: Optional[int] = None
    publisher: Optional[str] = None
    image_url: Optional[str] = None

    @validator('isbn')
    def validate_isbn(cls, v):
        if not validate_isbn(v):
            raise ValueError('Invalid ISBN format')
        return v.strip()

    @validator('title')
    def validate_title(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Title cannot be empty')
        if len(v) > ValidationLimits.MAX_QUERY_LENGTH:
            raise ValueError(f'Title too long (max {ValidationLimits.MAX_QUERY_LENGTH} characters)')
        return v.strip()

    @validator('author')
    def validate_author(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Author cannot be empty')
        return v.strip()

    @validator('year')
    def validate_year(cls, v):
        if v is not None:
            if not isinstance(v, int) or v < 1000 or v > 2024:
                raise ValueError('Year must be between 1000 and 2024')
        return v

    @validator('image_url')
    def validate_image_url(cls, v):
        if v and not v.startswith(('http://', 'https://')):
            raise ValueError('Image URL must start with http:// or https://')
        return v


class RatingModel(BaseModel):
    """Pydantic model for rating validation."""
    user_id: Union[str, int]
    isbn: str
    rating: Union[int, float]

    @validator('user_id')
    def validate_user_id(cls, v):
        return str(v).strip()

    @validator('isbn')
    def validate_isbn(cls, v):
        if not validate_isbn(v):
            raise ValueError('Invalid ISBN format')
        return v.strip()

    @validator('rating')
    def validate_rating(cls, v):
        if not ValidationLimits.MIN_RATING <= v <= ValidationLimits.MAX_RATING:
            raise ValueError(f'Rating must be between {ValidationLimits.MIN_RATING} and {ValidationLimits.MAX_RATING}')
        return float(v)


class UserModel(BaseModel):
    """Pydantic model for user validation."""
    user_id: Union[str, int]
    location: Optional[str] = None
    age: Optional[int] = None

    @validator('user_id')
    def validate_user_id(cls, v):
        return str(v).strip()

    @validator('age')
    def validate_age(cls, v):
        if v is not None:
            if not isinstance(v, int) or v < 13 or v > 120:
                raise ValueError('Age must be between 13 and 120')
        return v


class DataValidator:
    """Validates data quality and integrity."""

    def __init__(self):
        self.validation_errors = []

    def validate_books_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate books DataFrame."""
        logger.info("Validating books DataFrame")

        validation_report = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }

        # Check required columns
        required_columns = ['isbn', 'title', 'author']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            error_msg = f"Missing required columns: {missing_columns}"
            validation_report['errors'].append(error_msg)
            validation_report['valid'] = False
            return validation_report

        # Validate each book record
        invalid_records = 0
        validation_errors = []

        for idx, row in df.iterrows():
            try:
                BookModel(**row.to_dict())
            except PydanticValidationError as e:
                invalid_records += 1
                if len(validation_errors) < 10:  # Limit error reporting
                    validation_errors.append(f"Row {idx}: {str(e)}")

        if invalid_records > 0:
            validation_report['warnings'].append(f"{invalid_records} invalid book records found")
            validation_report['errors'].extend(validation_errors)

        # Check for duplicates
        duplicate_count = df.duplicated(subset=['isbn']).sum()
        if duplicate_count > 0:
            validation_report['warnings'].append(f"{duplicate_count} duplicate ISBN records found")

        # Data quality checks
        null_counts = df.isnull().sum()
        for column, null_count in null_counts.items():
            if null_count > 0:
                percentage = (null_count / len(df)) * 100
                validation_report['statistics'][f'{column}_null_percentage'] = percentage

                if percentage > 50:
                    validation_report['warnings'].append(
                        f"Column '{column}' has {percentage:.1f}% null values"
                    )

        logger.info(f"Books validation completed: {len(validation_report['errors'])} errors, {len(validation_report['warnings'])} warnings")
        return validation_report

    def validate_ratings_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate ratings DataFrame."""
        logger.info("Validating ratings DataFrame")

        validation_report = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }

        # Check required columns
        required_columns = ['user_id', 'isbn', 'rating']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            error_msg = f"Missing required columns: {missing_columns}"
            validation_report['errors'].append(error_msg)
            validation_report['valid'] = False
            return validation_report

        # Validate rating ranges
        invalid_ratings = df[
            (df['rating'] < ValidationLimits.MIN_RATING) | 
            (df['rating'] > ValidationLimits.MAX_RATING)
        ]

        if len(invalid_ratings) > 0:
            validation_report['warnings'].append(f"{len(invalid_ratings)} ratings outside valid range")

        # Check for missing values
        null_counts = df.isnull().sum()
        for column in required_columns:
            if null_counts[column] > 0:
                validation_report['errors'].append(f"Column '{column}' has {null_counts[column]} null values")
                validation_report['valid'] = False

        # Statistical validation
        validation_report['statistics'] = {
            'total_ratings': len(df),
            'unique_users': df['user_id'].nunique(),
            'unique_books': df['isbn'].nunique(),
            'rating_distribution': df['rating'].value_counts().to_dict(),
            'avg_rating': df['rating'].mean(),
            'rating_std': df['rating'].std()
        }

        # Check for unusual patterns
        user_rating_counts = df['user_id'].value_counts()
        if user_rating_counts.max() > 10000:
            validation_report['warnings'].append("Some users have unusually high number of ratings")

        book_rating_counts = df['isbn'].value_counts()
        if book_rating_counts.max() > 10000:
            validation_report['warnings'].append("Some books have unusually high number of ratings")

        logger.info(f"Ratings validation completed: {len(validation_report['errors'])} errors, {len(validation_report['warnings'])} warnings")
        return validation_report

    def validate_user_item_matrix(self, matrix: pd.DataFrame) -> Dict[str, Any]:
        """Validate user-item matrix for collaborative filtering."""
        logger.info("Validating user-item matrix")

        validation_report = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }

        # Check matrix dimensions
        rows, cols = matrix.shape
        validation_report['statistics'] = {
            'shape': (rows, cols),
            'density': (matrix > 0).sum().sum() / (rows * cols),
            'sparsity': 1 - ((matrix > 0).sum().sum() / (rows * cols))
        }

        # Check for empty rows/columns
        empty_users = (matrix == 0).all(axis=1).sum()
        empty_books = (matrix == 0).all(axis=0).sum()

        if empty_users > 0:
            validation_report['warnings'].append(f"{empty_users} users have no ratings")

        if empty_books > 0:
            validation_report['warnings'].append(f"{empty_books} books have no ratings")

        # Check sparsity
        sparsity = validation_report['statistics']['sparsity']
        if sparsity > 0.99:
            validation_report['warnings'].append(f"Matrix is very sparse ({sparsity:.2%})")

        logger.info(f"User-item matrix validation completed")
        return validation_report

    def validate_text_data(self, texts: List[str]) -> Dict[str, Any]:
        """Validate text data for content-based filtering."""
        logger.info(f"Validating {len(texts)} text entries")

        validation_report = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }

        # Check for empty texts
        empty_texts = sum(1 for text in texts if not text or not text.strip())
        if empty_texts > 0:
            validation_report['warnings'].append(f"{empty_texts} empty text entries found")

        # Text length statistics
        text_lengths = [len(text) for text in texts if text]
        if text_lengths:
            validation_report['statistics'] = {
                'total_texts': len(texts),
                'avg_length': np.mean(text_lengths),
                'min_length': min(text_lengths),
                'max_length': max(text_lengths),
                'std_length': np.std(text_lengths)
            }

            # Check for unusually short/long texts
            if min(text_lengths) < 10:
                validation_report['warnings'].append("Some texts are very short (< 10 characters)")

            if max(text_lengths) > ValidationLimits.MAX_DESCRIPTION_LENGTH:
                validation_report['warnings'].append(f"Some texts exceed maximum length ({ValidationLimits.MAX_DESCRIPTION_LENGTH} characters)")

        logger.info(f"Text validation completed")
        return validation_report

    def validate_model_input(self, data: Dict[str, Any], model_type: str) -> Dict[str, Any]:
        """Validate input data for model inference."""
        validation_report = {
            'valid': True,
            'errors': [],
            'warnings': []
        }

        if model_type == "collaborative_filtering":
            required_fields = ['user_id']

            # Check required fields
            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                validation_report['errors'].append(f"Missing required fields: {missing_fields}")
                validation_report['valid'] = False

            # Validate user_id format
            if 'user_id' in data:
                user_id = str(data['user_id']).strip()
                if not user_id:
                    validation_report['errors'].append("user_id cannot be empty")
                    validation_report['valid'] = False

        elif model_type == "content_based":
            required_fields = ['book_id']

            # Check required fields
            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                validation_report['errors'].append(f"Missing required fields: {missing_fields}")
                validation_report['valid'] = False

            # Validate book_id format
            if 'book_id' in data:
                book_id = str(data['book_id']).strip()
                if not book_id:
                    validation_report['errors'].append("book_id cannot be empty")
                    validation_report['valid'] = False

        elif model_type == "hybrid":
            required_fields = ['user_id', 'preferences']

            # Check required fields
            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                validation_report['errors'].append(f"Missing required fields: {missing_fields}")
                validation_report['valid'] = False

        # Validate recommendation count
        if 'num_recommendations' in data:
            num_recs = data['num_recommendations']
            if not isinstance(num_recs, int) or num_recs < ValidationLimits.MIN_RECOMMENDATIONS or num_recs > ValidationLimits.MAX_RECOMMENDATIONS:
                validation_report['errors'].append(
                    f"num_recommendations must be between {ValidationLimits.MIN_RECOMMENDATIONS} and {ValidationLimits.MAX_RECOMMENDATIONS}"
                )
                validation_report['valid'] = False

        return validation_report

    def generate_data_quality_report(self, df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        """Generate comprehensive data quality report."""
        logger.info(f"Generating data quality report for {dataset_name}")

        report = {
            'dataset_name': dataset_name,
            'timestamp': pd.Timestamp.now().isoformat(),
            'basic_info': {
                'total_records': len(df),
                'total_columns': len(df.columns),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
                'duplicate_records': df.duplicated().sum()
            },
            'column_analysis': {},
            'data_quality_score': 0.0,
            'recommendations': []
        }

        # Analyze each column
        total_score = 0
        for column in df.columns:
            column_analysis = self._analyze_column(df[column], column)
            report['column_analysis'][column] = column_analysis
            total_score += column_analysis['quality_score']

        # Calculate overall quality score
        report['data_quality_score'] = total_score / len(df.columns) if df.columns.any() else 0

        # Generate recommendations
        if report['data_quality_score'] < 0.7:
            report['recommendations'].append("Overall data quality is below recommended threshold")

        if report['basic_info']['duplicate_records'] > 0:
            report['recommendations'].append("Remove duplicate records")

        logger.info(f"Data quality report generated for {dataset_name}")
        return report

    def _analyze_column(self, series: pd.Series, column_name: str) -> Dict[str, Any]:
        """Analyze individual column quality."""
        analysis = {
            'data_type': str(series.dtype),
            'null_count': series.isnull().sum(),
            'null_percentage': (series.isnull().sum() / len(series)) * 100,
            'unique_count': series.nunique(),
            'quality_score': 1.0
        }

        # Reduce quality score based on null percentage
        if analysis['null_percentage'] > 50:
            analysis['quality_score'] -= 0.5
        elif analysis['null_percentage'] > 20:
            analysis['quality_score'] -= 0.3
        elif analysis['null_percentage'] > 10:
            analysis['quality_score'] -= 0.1

        # Add type-specific analysis
        if series.dtype in ['int64', 'float64']:
            analysis.update({
                'mean': series.mean(),
                'std': series.std(),
                'min': series.min(),
                'max': series.max(),
                'outliers': self._detect_outliers(series)
            })
        elif series.dtype == 'object':
            analysis.update({
                'avg_length': series.astype(str).str.len().mean(),
                'most_common': series.mode().iloc[0] if not series.mode().empty else None
            })

        return analysis

    def _detect_outliers(self, series: pd.Series) -> int:
        """Detect outliers using IQR method."""
        if series.dtype not in ['int64', 'float64']:
            return 0

        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = series[(series < lower_bound) | (series > upper_bound)]
        return len(outliers)


# Global validator instance
data_validator = DataValidator()
