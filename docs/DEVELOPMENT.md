# Development Guide

This guide covers setting up a development environment and contributing to the Book Recommender System.

## Development Setup

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- Git
- Code editor (VS Code recommended)

### Local Development Environment

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd book-recommender-system
   ```

2. **Set up Python environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements-dev.txt
   ```

3. **Set up external services**
   ```bash
   docker-compose up -d postgres redis ollama
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your local configuration
   ```

5. **Initialize data and models**
   ```bash
   python scripts/data_pipeline.py --download-sample
   python scripts/train_models.py --model all --no-ollama  # Use TF-IDF for faster development
   ```

6. **Run the application**
   ```bash
   python main.py
   ```

### VS Code Setup

Recommended extensions:
- Python
- Pylance
- Black Formatter
- isort
- Docker
- REST Client

**settings.json**:
```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.formatting.provider": "black",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests/"],
    "editor.formatOnSave": true,
    "[python]": {
        "editor.codeActionsOnSave": {
            "source.organizeImports": true
        }
    }
}
```

## Project Structure

```
book-recommender-system/
├── src/book_recommender/          # Main application code
│   ├── api/                       # API layer (routes, schemas)
│   ├── core/                      # Core utilities (config, logging)
│   ├── data/                      # Data handling (loading, processing)
│   ├── models/                    # ML models
│   ├── services/                  # Business logic services
│   └── utils/                     # Helper utilities
├── tests/                         # Test suites
│   ├── unit/                      # Unit tests
│   ├── integration/              # Integration tests
│   └── fixtures/                 # Test data and fixtures
├── data/                         # Data storage
│   ├── raw/                      # Raw datasets
│   ├── processed/                # Processed data
│   └── models/                   # Trained models
├── config/                       # Configuration files
├── docker/                       # Docker configurations
├── docs/                         # Documentation
├── scripts/                      # Utility scripts
└── monitoring/                   # Monitoring configurations
```

## Development Workflow

### 1. Feature Development

```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make changes
# ... code ...

# Run tests
pytest

# Run code quality checks
make lint
make format

# Commit changes
git add .
git commit -m "feat: add your feature description"

# Push and create PR
git push origin feature/your-feature-name
```

### 2. Code Quality

**Pre-commit hooks**:
```bash
# Install pre-commit
pip install pre-commit
pre-commit install

# Run manually
pre-commit run --all-files
```

**Code formatting**:
```bash
# Format code
black src/ tests/
isort src/ tests/

# Check formatting
black --check src/ tests/
isort --check-only src/ tests/
```

**Linting**:
```bash
# Run linter
flake8 src/ tests/

# Type checking
mypy src/
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/book_recommender --cov-report=html

# Run specific test file
pytest tests/unit/test_models.py

# Run tests with specific marker
pytest -m "not slow"
```

### Writing Tests

**Unit test example**:
```python
# tests/unit/test_collaborative_filtering.py
import pytest
from unittest.mock import Mock, patch
from book_recommender.models.collaborative_filtering import CollaborativeFilteringModel

class TestCollaborativeFilteringModel:
    def setup_method(self):
        self.model = CollaborativeFilteringModel()

    def test_model_initialization(self):
        assert self.model.n_neighbors == 6
        assert not self.model.is_trained

    @patch('book_recommender.data.data_loader.data_loader')
    def test_training(self, mock_loader):
        # Mock data
        mock_loader.load_books_data.return_value = Mock()
        mock_loader.load_ratings_data.return_value = Mock()

        # Test training
        stats = self.model.train(Mock(), Mock())

        assert self.model.is_trained
        assert 'total_books' in stats
```

**Integration test example**:
```python
# tests/integration/test_api.py
import pytest
from fastapi.testclient import TestClient
from book_recommender.main import create_app

@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)

def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_collaborative_recommendations(client):
    response = client.get("/api/v1/recommendations/collaborative/user123")
    assert response.status_code == 200
    data = response.json()
    assert "recommendations" in data
    assert data["user_id"] == "user123"
```

**Test fixtures**:
```python
# tests/fixtures/conftest.py
import pytest
import pandas as pd

@pytest.fixture
def sample_books_data():
    return pd.DataFrame({
        'isbn': ['ISBN001', 'ISBN002', 'ISBN003'],
        'title': ['Book 1', 'Book 2', 'Book 3'],
        'author': ['Author 1', 'Author 2', 'Author 3'],
        'year': [2020, 2021, 2022]
    })

@pytest.fixture
def sample_ratings_data():
    return pd.DataFrame({
        'user_id': ['user1', 'user2', 'user1'],
        'isbn': ['ISBN001', 'ISBN001', 'ISBN002'],
        'rating': [5, 4, 3]
    })
```

## Database Development

### Migrations with Alembic

```bash
# Create new migration
alembic revision --autogenerate -m "Add new table"

# Apply migrations
alembic upgrade head

# Downgrade to previous version
alembic downgrade -1

# Show migration history
alembic history
```

### Database Models

```python
# Example SQLAlchemy model
from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Book(Base):
    __tablename__ = 'books'

    id = Column(Integer, primary_key=True)
    isbn = Column(String(20), unique=True, nullable=False)
    title = Column(String(500), nullable=False)
    author = Column(String(200))
    year = Column(Integer)
    average_rating = Column(Float)
    created_at = Column(DateTime)
```

## API Development

### Adding New Endpoints

1. **Define Pydantic schemas**:
```python
# src/book_recommender/api/schemas.py
class NewFeatureRequest(BaseModel):
    parameter: str
    options: Optional[List[str]] = None

class NewFeatureResponse(BaseResponse):
    result: str
    data: Dict[str, Any]
```

2. **Implement service logic**:
```python
# src/book_recommender/services/new_service.py
class NewService:
    async def process_feature(self, request: NewFeatureRequest) -> Dict[str, Any]:
        # Business logic here
        return {"result": "success"}
```

3. **Add API routes**:
```python
# src/book_recommender/api/routes.py
@router.post("/new-feature", response_model=NewFeatureResponse)
async def new_feature_endpoint(
    request: NewFeatureRequest,
    service: NewService = Depends(get_new_service)
):
    result = await service.process_feature(request)
    return NewFeatureResponse(result=result["result"])
```

4. **Write tests**:
```python
# tests/unit/test_new_service.py
def test_new_service():
    service = NewService()
    request = NewFeatureRequest(parameter="test")
    result = await service.process_feature(request)
    assert result["result"] == "success"
```

## Model Development

### Adding New Models

1. **Create model class**:
```python
# src/book_recommender/models/new_model.py
class NewRecommendationModel:
    def __init__(self):
        self.is_trained = False

    def train(self, data):
        # Training logic
        self.is_trained = True
        return {"status": "success"}

    def predict(self, input_data):
        # Prediction logic
        return []
```

2. **Add to service layer**:
```python
# src/book_recommender/services/recommendation_service.py
from book_recommender.models.new_model import NewRecommendationModel

class RecommendationService:
    def __init__(self):
        self.new_model = NewRecommendationModel()

    async def get_new_recommendations(self, input_data):
        return self.new_model.predict(input_data)
```

3. **Update training scripts**:
```python
# scripts/train_models.py
async def train_new_model(args):
    # Training logic for new model
    pass
```

## Configuration Management

### Adding New Configuration

1. **Update config schemas**:
```python
# src/book_recommender/core/config.py
class NewModelConfig(BaseModel):
    parameter1: str = "default_value"
    parameter2: int = 100

class Settings(BaseSettings):
    new_model: NewModelConfig = NewModelConfig()
```

2. **Update YAML configs**:
```yaml
# config/config.yaml
new_model:
  parameter1: "production_value"
  parameter2: 200
```

## Debugging

### Local Debugging

```python
# Add breakpoints
import pdb; pdb.set_trace()

# Or use ipdb for better experience
import ipdb; ipdb.set_trace()
```

### Docker Debugging

```bash
# Debug inside container
docker-compose exec app python -c "
import pdb
from book_recommender.main import app
pdb.set_trace()
"

# Inspect container
docker-compose exec app /bin/bash
```

### Log Analysis

```python
# Add debug logging
import logging
logger = logging.getLogger(__name__)

def your_function():
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
```

## Performance Optimization

### Profiling

```python
# Profile code
import cProfile
import pstats

def profile_function():
    pr = cProfile.Profile()
    pr.enable()

    # Your code here

    pr.disable()
    stats = pstats.Stats(pr)
    stats.sort_stats('cumulative')
    stats.print_stats(10)
```

### Memory Monitoring

```python
# Monitor memory usage
import tracemalloc
import psutil

# Start tracing
tracemalloc.start()

# Your code here

# Get memory usage
current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage: {current / 1024 / 1024:.1f} MB")
print(f"Peak memory usage: {peak / 1024 / 1024:.1f} MB")
tracemalloc.stop()
```

## Contributing Guidelines

### Pull Request Process

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Update documentation
7. Submit a pull request

### Code Review Checklist

- [ ] Code follows project style guidelines
- [ ] Tests are included and pass
- [ ] Documentation is updated
- [ ] No breaking changes (or properly documented)
- [ ] Performance impact is considered
- [ ] Security implications are reviewed

### Commit Message Format

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Types: feat, fix, docs, style, refactor, test, chore

Examples:
- `feat(api): add collaborative filtering endpoint`
- `fix(models): handle empty datasets in training`
- `docs: update deployment guide`

## Useful Commands

```bash
# Start development environment
make dev

# Run all tests
make test

# Code quality checks
make lint
make format

# Build documentation
make docs

# Clean up
make clean

# Docker development
make docker-build
make docker-run
make docker-test
```

This development guide provides everything needed to contribute effectively to the Book Recommender System project.
