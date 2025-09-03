# API Documentation

## Overview

The Book Recommender System provides a RESTful API for generating book recommendations using collaborative filtering, content-based filtering, and hybrid approaches.

## Base URL

- Development: `http://localhost:8000/api/v1`
- Production: `https://your-domain.com/api/v1`

## Authentication

The API supports optional JWT-based authentication for enhanced features and rate limiting.

### Headers

```
Authorization: Bearer <jwt-token>
Content-Type: application/json
```

## Rate Limiting

- **Authenticated users**: 100 requests per minute
- **Anonymous users**: 60 requests per minute

Rate limit information is included in response headers:
- `X-RateLimit-Limit`: Request limit per window
- `X-RateLimit-Remaining`: Remaining requests in current window
- `X-RateLimit-Reset`: Time when the rate limit resets

## Endpoints

### Health Check

#### GET /health

Check the health status of the API.

**Response:**
```json
{
  "success": true,
  "status": "healthy",
  "checks": {
    "recommendation_service": {"status": "healthy"},
    "cache_service": {"status": "healthy"}
  },
  "version": "1.0.0",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

### Model Status

#### GET /models/status

Get the status of all recommendation models.

**Response:**
```json
{
  "success": true,
  "collaborative_filtering": {
    "status": "trained",
    "total_books": 1000,
    "total_users": 500,
    "sparsity": 0.95
  },
  "content_based": {
    "status": "trained",
    "total_books": 7000,
    "embedding_method": "ollama"
  },
  "hybrid": {
    "status": "trained"
  }
}
```

### Recommendations

#### GET /recommendations/collaborative/{user_id}

Get collaborative filtering recommendations for a user.

**Parameters:**
- `user_id` (path, required): User identifier
- `num_recommendations` (query, optional): Number of recommendations (default: 10, max: 50)

**Example Request:**
```
GET /recommendations/collaborative/user123?num_recommendations=5
```

**Response:**
```json
{
  "success": true,
  "user_id": "user123",
  "model_type": "collaborative_filtering",
  "recommendations": [
    {
      "title": "The Great Gatsby",
      "authors": "F. Scott Fitzgerald",
      "predicted_rating": 4.2,
      "rank": 1
    }
  ],
  "total_count": 5
}
```

#### GET /recommendations/content/{book_id}

Get content-based recommendations for a book.

**Parameters:**
- `book_id` (path, required): Book identifier
- `num_recommendations` (query, optional): Number of recommendations (default: 10, max: 50)

**Example Request:**
```
GET /recommendations/content/123?num_recommendations=5
```

**Response:**
```json
{
  "success": true,
  "book_id": 123,
  "book_details": {
    "title": "Sample Book",
    "authors": "Author Name",
    "categories": "Fiction"
  },
  "model_type": "content_based",
  "recommendations": [
    {
      "title": "Similar Book",
      "authors": "Another Author",
      "similarity_score": 0.85,
      "rank": 1
    }
  ],
  "total_count": 5
}
```

#### POST /recommendations/hybrid

Get hybrid recommendations combining collaborative and content-based approaches.

**Request Body:**
```json
{
  "user_id": "user123",
  "book_id": 456,
  "preferences": {
    "genres": ["Fiction", "Mystery"],
    "authors": ["Agatha Christie"],
    "keywords": ["detective", "crime"]
  },
  "num_recommendations": 10
}
```

**Response:**
```json
{
  "success": true,
  "model_type": "hybrid",
  "recommendations": [
    {
      "title": "Murder on the Orient Express",
      "authors": "Agatha Christie",
      "hybrid_score": 0.92,
      "collaborative_score": 0.88,
      "content_score": 0.95,
      "rank": 1
    }
  ],
  "total_count": 10
}
```

### Books

#### GET /books/search

Search for books by text query.

**Parameters:**
- `query` (query, required): Search query
- `num_results` (query, optional): Number of results (default: 20, max: 100)

**Example Request:**
```
GET /books/search?query=science fiction&num_results=10
```

**Response:**
```json
{
  "success": true,
  "query": "science fiction",
  "results": [
    {
      "book_id": 789,
      "title": "Dune",
      "authors": "Frank Herbert",
      "categories": "Science Fiction",
      "relevance_score": 0.95
    }
  ],
  "total_count": 10
}
```

#### GET /books/{book_id}

Get detailed information about a book.

**Parameters:**
- `book_id` (path, required): Book identifier

**Response:**
```json
{
  "success": true,
  "book": {
    "book_id": 789,
    "title": "Dune",
    "authors": "Frank Herbert",
    "categories": "Science Fiction",
    "description": "Epic science fiction novel...",
    "published_year": 1965,
    "average_rating": 4.2,
    "num_pages": 688
  }
}
```

#### POST /books/similar

Find books similar to a given book title.

**Request Body:**
```json
{
  "book_title": "The Great Gatsby",
  "num_recommendations": 5
}
```

**Response:**
```json
{
  "success": true,
  "book_title": "The Great Gatsby",
  "model_type": "hybrid_similarity",
  "recommendations": [
    {
      "title": "The Catcher in the Rye",
      "authors": "J.D. Salinger",
      "similarity_score": 0.78,
      "rank": 1
    }
  ],
  "total_count": 5
}
```

#### GET /books/popular

Get popular books.

**Parameters:**
- `num_books` (query, optional): Number of books (default: 20, max: 100)

**Response:**
```json
{
  "success": true,
  "model_type": "popularity",
  "books": [
    {
      "title": "To Kill a Mockingbird",
      "authors": "Harper Lee",
      "popularity_score": 98.5,
      "num_ratings": 15000,
      "avg_rating": 4.3,
      "rank": 1
    }
  ],
  "total_count": 20
}
```

## Error Handling

All API endpoints return consistent error responses:

```json
{
  "success": false,
  "error_code": "VALIDATION_ERROR",
  "message": "Request validation failed",
  "details": {
    "validation_errors": [
      {
        "field": "user_id",
        "message": "User ID cannot be empty"
      }
    ]
  },
  "timestamp": "2024-01-01T00:00:00Z"
}
```

### Error Codes

- `VALIDATION_ERROR` (422): Request validation failed
- `NOT_FOUND` (404): Resource not found
- `UNAUTHORIZED` (401): Authentication required
- `FORBIDDEN` (403): Insufficient permissions
- `RATE_LIMITED` (429): Rate limit exceeded
- `SERVICE_UNAVAILABLE` (503): Service temporarily unavailable
- `INTERNAL_ERROR` (500): Internal server error

## SDKs and Libraries

### Python SDK

```python
from book_recommender_client import BookRecommenderClient

client = BookRecommenderClient(
    base_url="http://localhost:8000/api/v1",
    api_key="your-api-key"
)

# Get collaborative recommendations
recommendations = client.get_collaborative_recommendations(
    user_id="user123",
    num_recommendations=10
)

# Get content-based recommendations
recommendations = client.get_content_recommendations(
    book_id=456,
    num_recommendations=10
)

# Search books
results = client.search_books(
    query="science fiction",
    num_results=20
)
```

### JavaScript SDK

```javascript
import { BookRecommenderClient } from 'book-recommender-js';

const client = new BookRecommenderClient({
  baseURL: 'http://localhost:8000/api/v1',
  apiKey: 'your-api-key'
});

// Get hybrid recommendations
const recommendations = await client.getHybridRecommendations({
  userId: 'user123',
  preferences: {
    genres: ['Fiction', 'Mystery']
  },
  numRecommendations: 10
});
```

## Webhooks

The API supports webhooks for real-time notifications about recommendation updates and system events.

### Webhook Events

- `recommendation.generated`: New recommendations generated
- `model.trained`: Model training completed
- `system.health_check_failed`: System health check failed

### Webhook Payload

```json
{
  "event": "recommendation.generated",
  "timestamp": "2024-01-01T00:00:00Z",
  "data": {
    "user_id": "user123",
    "model_type": "hybrid",
    "num_recommendations": 10
  }
}
```

## Testing

You can test the API using curl, Postman, or any HTTP client:

```bash
# Health check
curl -X GET http://localhost:8000/health

# Get collaborative recommendations
curl -X GET http://localhost:8000/api/v1/recommendations/collaborative/user123

# Search books
curl -X GET "http://localhost:8000/api/v1/books/search?query=python&num_results=5"

# Get hybrid recommendations
curl -X POST http://localhost:8000/api/v1/recommendations/hybrid \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user123", "num_recommendations": 5}'
```

## OpenAPI Specification

The complete OpenAPI 3.0 specification is available at:
- Interactive docs: `http://localhost:8000/docs`
- JSON schema: `http://localhost:8000/openapi.json`
