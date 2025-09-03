# 📚 Book Recommender System

A production-ready hybrid book recommendation system combining collaborative filtering and content-based approaches to suggest books based on user preferences and textual content.

## 🚀 Features

- **🤝 Collaborative Filtering**: Uses K-Nearest Neighbors (KNN) to recommend books based on user ratings
- **📝 Content-Based Recommendations**: Leverages text embeddings and FAISS for efficient similarity search
- **🔄 Hybrid Model**: Combines both approaches for better recommendations
- **🌐 RESTful API**: FastAPI-based API with automatic OpenAPI documentation
- **🐳 Docker Support**: Containerized deployment with Docker and Docker Compose
- **📊 Monitoring**: Built-in health checks and performance monitoring
- **🔒 Security**: Input validation, rate limiting, and authentication
- **⚡ Performance**: Caching layer and optimized algorithms
- **🧪 Testing**: Comprehensive unit and integration tests
- **📖 Documentation**: Complete API and deployment documentation

## 🏗️ Architecture

```
book-recommender-system/
├── src/book_recommender/        # Main application code
│   ├── api/                     # API routes and schemas
│   ├── core/                    # Core configuration and utilities
│   ├── data/                    # Data loading and preprocessing
│   ├── models/                  # ML models and algorithms
│   ├── services/                # Business logic services
│   └── utils/                   # Helper utilities
├── tests/                       # Test suites
├── config/                      # Configuration files
├── docker/                      # Docker configurations
├── scripts/                     # Utility scripts
└── docs/                        # Documentation
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Docker (optional)
- Ollama (for content-based recommendations)

### Local Development

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd book-recommender-system
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements-dev.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Prepare data**
   ```bash
   # Place your datasets in data/raw/
   python scripts/data_pipeline.py
   ```

6. **Train models**
   ```bash
   python scripts/train_models.py
   ```

7. **Run the application**
   ```bash
   python main.py
   ```

8. **Access the API**
   - API: http://localhost:8000
   - Documentation: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health

### Docker Deployment

1. **Build and run with Docker Compose**
   ```bash
   docker-compose up --build
   ```

2. **For production deployment**
   ```bash
   docker-compose -f docker-compose.prod.yml up --build
   ```

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/recommendations/collaborative/{user_id}` | GET | Collaborative filtering recommendations |
| `/recommendations/content/{book_id}` | GET | Content-based recommendations |
| `/recommendations/hybrid/{user_id}` | POST | Hybrid recommendations |
| `/books/search` | GET | Search books |
| `/books/{book_id}` | GET | Get book details |

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/book_recommender

# Run specific test types
pytest tests/unit/
pytest tests/integration/
```

## 📈 Performance

The system includes several performance optimizations:

- **Caching**: Redis-based caching for frequent requests
- **Batch Processing**: Efficient batch recommendation generation
- **Async Operations**: FastAPI async support for concurrent requests
- **Database Optimization**: Optimized queries and indexing
- **Model Optimization**: Pre-computed similarity matrices

## 🔧 Configuration

Configuration is managed through YAML files in the `config/` directory:

- `config.yaml`: Base configuration
- `config.development.yaml`: Development overrides
- `config.production.yaml`: Production overrides

## 📚 Documentation

- [API Documentation](docs/API.md)
- [Architecture Overview](docs/ARCHITECTURE.md)
- [Deployment Guide](docs/DEPLOYMENT.md)
- [Development Guide](docs/DEVELOPMENT.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Book-Crossing Dataset
- FAISS for similarity search
- FastAPI for the web framework
- Streamlit for the original UI inspiration
