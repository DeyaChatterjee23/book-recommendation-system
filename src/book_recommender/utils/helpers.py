"""Helper utilities for the Book Recommender System."""

import asyncio
import hashlib
import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

import pandas as pd
import numpy as np
from pydantic import BaseModel


def serialize_datetime(dt: datetime) -> str:
    """Serialize datetime to ISO format string."""
    return dt.isoformat()


def deserialize_datetime(dt_str: str) -> datetime:
    """Deserialize ISO format string to datetime."""
    return datetime.fromisoformat(dt_str)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero."""
    return numerator / denominator if denominator != 0 else default


def normalize_text(text: str) -> str:
    """Normalize text for consistent processing."""
    if not text:
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^a-zA-Z0-9\s.,;:!?-]', '', text)

    return text.strip()


def generate_hash(data: Union[str, Dict, List]) -> str:
    """Generate MD5 hash for data."""
    if isinstance(data, (dict, list)):
        data = json.dumps(data, sort_keys=True)

    return hashlib.md5(data.encode('utf-8')).hexdigest()


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split a list into chunks of specified size."""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def flatten_list(nested_list: List[List[Any]]) -> List[Any]:
    """Flatten a nested list."""
    return [item for sublist in nested_list for item in sublist]


def safe_get_nested(data: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    """Safely get nested dictionary value."""
    for key in keys:
        if isinstance(data, dict) and key in data:
            data = data[key]
        else:
            return default
    return data


def calculate_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def validate_isbn(isbn: str) -> bool:
    """Validate ISBN format."""
    # Remove hyphens and spaces
    isbn = re.sub(r'[\s-]', '', isbn)

    # Check if it's a valid 10 or 13 digit ISBN
    if len(isbn) == 10:
        return validate_isbn10(isbn)
    elif len(isbn) == 13:
        return validate_isbn13(isbn)

    return False


def validate_isbn10(isbn: str) -> bool:
    """Validate ISBN-10 format."""
    if len(isbn) != 10:
        return False

    total = 0
    for i, digit in enumerate(isbn[:-1]):
        if not digit.isdigit():
            return False
        total += int(digit) * (10 - i)

    check_digit = isbn[-1]
    if check_digit == 'X':
        total += 10
    elif check_digit.isdigit():
        total += int(check_digit)
    else:
        return False

    return total % 11 == 0


def validate_isbn13(isbn: str) -> bool:
    """Validate ISBN-13 format."""
    if len(isbn) != 13:
        return False

    if not isbn.isdigit():
        return False

    total = 0
    for i, digit in enumerate(isbn[:-1]):
        weight = 1 if i % 2 == 0 else 3
        total += int(digit) * weight

    check_digit = int(isbn[-1])
    calculated_check = (10 - (total % 10)) % 10

    return check_digit == calculated_check


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    if size_bytes == 0:
        return "0 B"

    size_names = ["B", "KB", "MB", "GB", "TB"]
    size_index = 0

    while size_bytes >= 1024 and size_index < len(size_names) - 1:
        size_bytes /= 1024.0
        size_index += 1

    return f"{size_bytes:.1f} {size_names[size_index]}"


def ensure_directory(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if it doesn't."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


class Timer:
    """Context manager for timing operations."""

    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = datetime.now()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        print(f"{self.name} completed in {duration:.2f} seconds")

    @property
    def duration(self) -> float:
        """Get duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0


async def retry_async(
    func,
    max_retries: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """Retry an async function with exponential backoff."""

    for attempt in range(max_retries):
        try:
            return await func()
        except exceptions as e:
            if attempt == max_retries - 1:
                raise e

            wait_time = delay * (backoff_factor ** attempt)
            await asyncio.sleep(wait_time)


def pydantic_to_dict(model: BaseModel, exclude_none: bool = True) -> Dict[str, Any]:
    """Convert Pydantic model to dictionary."""
    return model.model_dump(exclude_none=exclude_none)


def dict_to_pydantic(data: Dict[str, Any], model_class: type) -> BaseModel:
    """Convert dictionary to Pydantic model."""
    return model_class(**data)
