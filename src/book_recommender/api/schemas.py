"""Pydantic schemas for API request/response models."""

from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field, validator

from book_recommender.utils.constants import ValidationLimits


# Base response models
class BaseResponse(BaseModel):
    """Base response model."""
    success: bool = True
    message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ErrorResponse(BaseResponse):
    """Error response model."""
    success: bool = False
    error_code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


# Book models
class BookDetails(BaseModel):
    """Book details model."""
    book_id: Optional[int] = None
    title: str
    authors: Optional[str] = None
    categories: Optional[str] = None
    description: Optional[str] = None
    published_year: Optional[int] = None
    average_rating: Optional[float] = None
    num_pages: Optional[int] = None
    ratings_count: Optional[int] = None
    isbn13: Optional[str] = None
    isbn10: Optional[str] = None
    image_url: Optional[str] = None


class BookRecommendation(BaseModel):
    """Book recommendation model."""
    title: str
    authors: Optional[str] = None
    categories: Optional[str] = None
    book_id: Optional[int] = None
    rank: int
    similarity_score: Optional[float] = None
    predicted_rating: Optional[float] = None
    popularity_score: Optional[float] = None
    hybrid_score: Optional[float] = None
    collaborative_score: Optional[float] = None
    content_score: Optional[float] = None
    average_rating: Optional[float] = None
    published_year: Optional[int] = None


class SearchResult(BaseModel):
    """Search result model."""
    book_id: int
    title: str
    authors: Optional[str] = None
    categories: Optional[str] = None
    relevance_score: float
    average_rating: Optional[float] = None
    published_year: Optional[int] = None


# Request models
class CollaborativeRecommendationRequest(BaseModel):
    """Request model for collaborative filtering recommendations."""
    user_id: str = Field(..., description="User ID for recommendations")
    num_recommendations: int = Field(default=10, ge=ValidationLimits.MIN_RECOMMENDATIONS, le=ValidationLimits.MAX_RECOMMENDATIONS)

    @validator('user_id')
    def validate_user_id(cls, v):
        if not v or not v.strip():
            raise ValueError('User ID cannot be empty')
        return v.strip()


class ContentRecommendationRequest(BaseModel):
    """Request model for content-based recommendations."""
    book_id: int = Field(..., description="Book ID for recommendations")
    num_recommendations: int = Field(default=10, ge=ValidationLimits.MIN_RECOMMENDATIONS, le=ValidationLimits.MAX_RECOMMENDATIONS)

    @validator('book_id')
    def validate_book_id(cls, v):
        if v < 0:
            raise ValueError('Book ID must be non-negative')
        return v


class UserPreferences(BaseModel):
    """User preferences model for hybrid recommendations."""
    genres: Optional[List[str]] = None
    authors: Optional[List[str]] = None
    keywords: Optional[List[str]] = None
    description: Optional[str] = None
    min_rating: Optional[float] = Field(None, ge=0, le=10)
    max_rating: Optional[float] = Field(None, ge=0, le=10)
    year_range: Optional[Dict[str, int]] = None

    @validator('description')
    def validate_description(cls, v):
        if v and len(v) > ValidationLimits.MAX_DESCRIPTION_LENGTH:
            raise ValueError(f'Description too long (max {ValidationLimits.MAX_DESCRIPTION_LENGTH} characters)')
        return v

    @validator('year_range')
    def validate_year_range(cls, v):
        if v:
            if 'min' in v and 'max' in v:
                if v['min'] > v['max']:
                    raise ValueError('Min year cannot be greater than max year')
        return v


class HybridRecommendationRequest(BaseModel):
    """Request model for hybrid recommendations."""
    user_id: Optional[str] = None
    book_id: Optional[int] = None
    preferences: Optional[UserPreferences] = None
    num_recommendations: int = Field(default=10, ge=ValidationLimits.MIN_RECOMMENDATIONS, le=ValidationLimits.MAX_RECOMMENDATIONS)

    @validator('user_id')
    def validate_user_id(cls, v):
        if v is not None and (not v or not v.strip()):
            raise ValueError('User ID cannot be empty if provided')
        return v.strip() if v else v

    @validator('book_id')
    def validate_book_id(cls, v):
        if v is not None and v < 0:
            raise ValueError('Book ID must be non-negative if provided')
        return v

    def __init__(self, **data):
        super().__init__(**data)
        # Validate that at least one input is provided
        if not any([self.user_id, self.book_id, self.preferences]):
            raise ValueError('At least one of user_id, book_id, or preferences must be provided')


class BookSearchRequest(BaseModel):
    """Request model for book search."""
    query: str = Field(..., description="Search query")
    num_results: int = Field(default=20, ge=1, le=100)

    @validator('query')
    def validate_query(cls, v):
        if not v or not v.strip():
            raise ValueError('Search query cannot be empty')
        if len(v) > ValidationLimits.MAX_QUERY_LENGTH:
            raise ValueError(f'Query too long (max {ValidationLimits.MAX_QUERY_LENGTH} characters)')
        return v.strip()


class SimilarBooksRequest(BaseModel):
    """Request model for finding similar books."""
    book_title: str = Field(..., description="Title of the book to find similar books for")
    num_recommendations: int = Field(default=10, ge=ValidationLimits.MIN_RECOMMENDATIONS, le=ValidationLimits.MAX_RECOMMENDATIONS)

    @validator('book_title')
    def validate_book_title(cls, v):
        if not v or not v.strip():
            raise ValueError('Book title cannot be empty')
        return v.strip()


# Response models
class CollaborativeRecommendationResponse(BaseResponse):
    """Response model for collaborative filtering recommendations."""
    user_id: str
    model_type: str
    recommendations: List[BookRecommendation]
    total_count: int


class ContentRecommendationResponse(BaseResponse):
    """Response model for content-based recommendations."""
    book_id: int
    book_details: BookDetails
    model_type: str
    recommendations: List[BookRecommendation]
    total_count: int


class HybridRecommendationResponse(BaseResponse):
    """Response model for hybrid recommendations."""
    model_type: str
    input_data: Dict[str, Any]
    recommendations: List[BookRecommendation]
    total_count: int


class BookSearchResponse(BaseResponse):
    """Response model for book search."""
    query: str
    results: List[SearchResult]
    total_count: int


class PopularBooksResponse(BaseResponse):
    """Response model for popular books."""
    model_type: str
    books: List[BookRecommendation]
    total_count: int


class BookDetailsResponse(BaseResponse):
    """Response model for book details."""
    book: BookDetails


class SimilarBooksResponse(BaseResponse):
    """Response model for similar books."""
    book_title: str
    model_type: str
    recommendations: List[BookRecommendation]
    total_count: int


# System models
class ModelInfo(BaseModel):
    """Model information."""
    status: str
    model_type: Optional[str] = None
    total_books: Optional[int] = None
    total_users: Optional[int] = None
    embedding_method: Optional[str] = None
    embedding_dimension: Optional[int] = None
    sparsity: Optional[float] = None
    parameters: Optional[Dict[str, Any]] = None


class HealthCheckResponse(BaseResponse):
    """Health check response model."""
    status: str
    checks: Dict[str, Any]
    version: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ModelStatusResponse(BaseResponse):
    """Model status response."""
    collaborative_filtering: ModelInfo
    content_based: ModelInfo
    hybrid: ModelInfo
    cache_service: Dict[str, Any]
    system: Dict[str, Any]


class CacheInfo(BaseModel):
    """Cache service information."""
    status: str
    redis_version: Optional[str] = None
    used_memory_human: Optional[str] = None
    connected_clients: Optional[int] = None
    keyspace_hits: Optional[int] = None
    keyspace_misses: Optional[int] = None


class CacheOperationResponse(BaseResponse):
    """Cache operation response."""
    operation: str
    result: Dict[str, Any]


# Batch operation models
class BatchRecommendationRequest(BaseModel):
    """Batch recommendation request."""
    requests: List[Union[CollaborativeRecommendationRequest, ContentRecommendationRequest, HybridRecommendationRequest]]

    @validator('requests')
    def validate_requests(cls, v):
        if len(v) == 0:
            raise ValueError('At least one request must be provided')
        if len(v) > 100:  # Reasonable limit for batch operations
            raise ValueError('Too many requests in batch (max 100)')
        return v


class BatchRecommendationResponse(BaseResponse):
    """Batch recommendation response."""
    results: List[Union[CollaborativeRecommendationResponse, ContentRecommendationResponse, HybridRecommendationResponse]]
    successful: int
    failed: int
    errors: List[str] = []


# Training and management models
class TrainingRequest(BaseModel):
    """Model training request."""
    model_type: str = Field(..., regex="^(collaborative|content|hybrid)$")
    force_retrain: bool = False
    use_ollama: bool = True

    class Config:
        schema_extra = {
            "example": {
                "model_type": "hybrid",
                "force_retrain": False,
                "use_ollama": True
            }
        }


class TrainingResponse(BaseResponse):
    """Model training response."""
    model_type: str
    training_stats: Dict[str, Any]
    duration_seconds: float


class CacheWarmupRequest(BaseModel):
    """Cache warmup request."""
    sample_requests: List[Dict[str, Any]]

    @validator('sample_requests')
    def validate_sample_requests(cls, v):
        if len(v) == 0:
            raise ValueError('At least one sample request must be provided')
        return v


class CacheWarmupResponse(BaseResponse):
    """Cache warmup response."""
    total_requests: int
    successful: int
    failed: int
    duration_seconds: float
