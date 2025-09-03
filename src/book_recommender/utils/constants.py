"""Constants for the Book Recommender System."""

# API Response codes
class ResponseCodes:
    SUCCESS = "SUCCESS"
    ERROR = "ERROR"
    NOT_FOUND = "NOT_FOUND"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    UNAUTHORIZED = "UNAUTHORIZED"
    RATE_LIMITED = "RATE_LIMITED"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"


# Model types
class ModelTypes:
    COLLABORATIVE_FILTERING = "collaborative_filtering"
    CONTENT_BASED = "content_based"
    HYBRID = "hybrid"


# Recommendation types
class RecommendationTypes:
    USER_BASED = "user_based"
    ITEM_BASED = "item_based"
    CONTENT_BASED = "content_based"
    HYBRID = "hybrid"


# Cache keys
class CacheKeys:
    USER_RECOMMENDATIONS = "user_recommendations:{user_id}"
    BOOK_RECOMMENDATIONS = "book_recommendations:{book_id}"
    BOOK_DETAILS = "book_details:{book_id}"
    POPULAR_BOOKS = "popular_books"
    SIMILARITY_MATRIX = "similarity_matrix"


# Data validation limits
class ValidationLimits:
    MAX_RECOMMENDATIONS = 100
    MIN_RECOMMENDATIONS = 1
    MAX_QUERY_LENGTH = 1000
    MAX_DESCRIPTION_LENGTH = 5000
    MIN_RATING = 0
    MAX_RATING = 10


# File paths
class FilePaths:
    RAW_DATA = "data/raw"
    PROCESSED_DATA = "data/processed"
    MODELS = "data/models"
    LOGS = "logs"
    CONFIG = "config"


# HTTP status messages
class StatusMessages:
    SUCCESS = "Operation completed successfully"
    CREATED = "Resource created successfully"
    UPDATED = "Resource updated successfully"
    DELETED = "Resource deleted successfully"
    NOT_FOUND = "Resource not found"
    VALIDATION_ERROR = "Validation failed"
    UNAUTHORIZED = "Authentication required"
    FORBIDDEN = "Access denied"
    RATE_LIMITED = "Rate limit exceeded"
    SERVICE_UNAVAILABLE = "Service temporarily unavailable"
    INTERNAL_ERROR = "An internal error occurred"


# Model parameters
class ModelDefaults:
    COLLABORATIVE_FILTERING = {
        "n_neighbors": 6,
        "algorithm": "brute",
        "min_user_ratings": 200,
        "min_book_ratings": 50
    }

    CONTENT_BASED = {
        "embedding_dim": 2304,
        "similarity_metric": "cosine",
        "max_features": 10000
    }

    HYBRID = {
        "collaborative_weight": 0.6,
        "content_weight": 0.4
    }


# Regular expressions for validation
class RegexPatterns:
    ISBN = r"^(?:ISBN(?:-1[03])?:? )?(?=[0-9X]{10}$|(?=(?:[0-9]+[- ]){3})[- 0-9X]{13}$|97[89][0-9]{10}$|(?=(?:[0-9]+[- ]){4})[- 0-9]{17}$)(?:97[89][- ]?)?[0-9]{1,5}[- ]?[0-9]+[- ]?[0-9]+[- ]?[0-9X]$"
    EMAIL = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    USERNAME = r"^[a-zA-Z0-9_]{3,20}$"


# Error messages
class ErrorMessages:
    BOOK_NOT_FOUND = "Book not found"
    USER_NOT_FOUND = "User not found"
    INVALID_BOOK_ID = "Invalid book ID"
    INVALID_USER_ID = "Invalid user ID"
    INVALID_RATING = "Rating must be between {min_rating} and {max_rating}"
    MODEL_NOT_LOADED = "Recommendation model not loaded"
    INSUFFICIENT_DATA = "Insufficient data for recommendations"
    EXTERNAL_SERVICE_ERROR = "External service unavailable"
    INVALID_INPUT = "Invalid input data"
    RATE_LIMIT_EXCEEDED = "Rate limit exceeded. Please try again later."
