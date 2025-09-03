# Deployment Guide

This guide covers deploying the Book Recommender System in various environments.

## Prerequisites

- Docker 20.10+ and Docker Compose 2.0+
- Python 3.11+ (for local development)
- 8GB+ RAM and 4+ CPU cores recommended
- PostgreSQL 15+ (for production)
- Redis 7+ (for production)

## Quick Start

### 1. Local Development with Docker Compose

```bash
# Clone the repository
git clone <repository-url>
cd book-recommender-system

# Copy environment file
cp .env.example .env

# Edit configuration
nano .env

# Start all services
docker-compose up -d

# Check health
curl http://localhost:8000/health

# View logs
docker-compose logs -f app
```

### 2. Manual Setup (Development)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements-dev.txt

# Set up environment
cp .env.example .env

# Start external services
docker-compose up -d postgres redis ollama

# Run data pipeline
python scripts/data_pipeline.py --download-sample

# Train models
python scripts/train_models.py --model all

# Start application
python main.py
```

## Production Deployment

### Option 1: Docker Compose (Single Server)

```bash
# Clone repository on production server
git clone <repository-url>
cd book-recommender-system

# Create production environment file
cp .env.example .env.production
nano .env.production

# Set production values
export ENVIRONMENT=production
export SECRET_KEY=$(openssl rand -hex 32)
export DATABASE_URL=postgresql://user:pass@localhost:5432/bookdb
export REDIS_URL=redis://:password@localhost:6379/0

# Deploy with production compose file
docker-compose -f docker-compose.prod.yml up -d

# Initialize database and train models
docker-compose -f docker-compose.prod.yml exec app python scripts/data_pipeline.py
docker-compose -f docker-compose.prod.yml exec app python scripts/train_models.py
```

### Option 2: Kubernetes Deployment

#### Prerequisites
- Kubernetes cluster (1.24+)
- kubectl configured
- Helm 3.0+ (optional)

#### Deploy with Kubernetes manifests

```bash
# Create namespace
kubectl create namespace book-recommender

# Create config maps and secrets
kubectl create configmap app-config --from-env-file=.env.production -n book-recommender
kubectl create secret generic app-secrets --from-env-file=.env.secrets -n book-recommender

# Deploy PostgreSQL
kubectl apply -f k8s/postgres.yaml -n book-recommender

# Deploy Redis
kubectl apply -f k8s/redis.yaml -n book-recommender

# Deploy application
kubectl apply -f k8s/app.yaml -n book-recommender

# Deploy ingress
kubectl apply -f k8s/ingress.yaml -n book-recommender

# Check deployment
kubectl get pods -n book-recommender
kubectl get services -n book-recommender
```

#### Deploy with Helm (recommended)

```bash
# Add repository
helm repo add book-recommender ./helm

# Install
helm install book-recommender book-recommender/book-recommender \
  --namespace book-recommender \
  --create-namespace \
  --values values.production.yaml

# Upgrade
helm upgrade book-recommender book-recommender/book-recommender \
  --namespace book-recommender \
  --values values.production.yaml
```

### Option 3: Cloud Deployment

#### AWS ECS with Fargate

```bash
# Build and push Docker image
docker build -t book-recommender:latest .
docker tag book-recommender:latest <account-id>.dkr.ecr.<region>.amazonaws.com/book-recommender:latest
docker push <account-id>.dkr.ecr.<region>.amazonaws.com/book-recommender:latest

# Deploy using AWS CLI
aws ecs create-service \
  --cluster book-recommender-cluster \
  --service-name book-recommender-service \
  --task-definition book-recommender:1 \
  --desired-count 2 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx],assignPublicIp=ENABLED}"
```

#### Google Cloud Run

```bash
# Build and deploy
gcloud builds submit --tag gcr.io/PROJECT-ID/book-recommender
gcloud run deploy book-recommender \
  --image gcr.io/PROJECT-ID/book-recommender \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --concurrency 100
```

## Database Setup

### PostgreSQL Configuration

```sql
-- Create database
CREATE DATABASE bookdb;
CREATE USER bookdb_user WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE bookdb TO bookdb_user;

-- Enable extensions
\c bookdb;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
```

### Migration Scripts

```bash
# Run database migrations
docker-compose exec app alembic upgrade head

# Create migration (if needed)
docker-compose exec app alembic revision --autogenerate -m "description"
```

## Model Training

### Initial Training

```bash
# Download and process data
python scripts/data_pipeline.py --download-sample --split-data

# Train all models
python scripts/train_models.py --model all

# Verify models
curl http://localhost:8000/api/v1/models/status
```

### Scheduled Retraining

Add to crontab for automatic retraining:

```bash
# Retrain models daily at 2 AM
0 2 * * * cd /opt/book-recommender && python scripts/train_models.py --model all > logs/training.log 2>&1
```

Or use Kubernetes CronJob:

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: model-training
spec:
  schedule: "0 2 * * *"
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: trainer
            image: book-recommender:latest
            command: ["python", "scripts/train_models.py", "--model", "all"]
          restartPolicy: OnFailure
```

## Monitoring Setup

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'book-recommender'
    static_configs:
      - targets: ['app:8000']
    metrics_path: '/metrics'
```

### Grafana Dashboards

Import pre-built dashboards:

```bash
# Copy dashboard configurations
cp monitoring/grafana/dashboards/*.json /var/lib/grafana/dashboards/

# Restart Grafana
docker-compose restart grafana
```

## Load Balancing

### Nginx Configuration

```nginx
upstream book_recommender {
    server app1:8000;
    server app2:8000;
    server app3:8000;
}

server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://book_recommender;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /health {
        proxy_pass http://book_recommender;
        access_log off;
    }
}
```

## Security Configuration

### SSL/TLS Setup

```bash
# Generate SSL certificate with Let's Encrypt
certbot --nginx -d your-domain.com

# Update nginx configuration for HTTPS
server {
    listen 443 ssl;
    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;
    # ... rest of configuration
}
```

### Firewall Configuration

```bash
# Ubuntu/Debian with UFW
ufw allow 22/tcp
ufw allow 80/tcp
ufw allow 443/tcp
ufw enable

# Block direct access to database
ufw deny 5432/tcp
ufw deny 6379/tcp
```

## Backup and Recovery

### Database Backup

```bash
# Manual backup
docker-compose exec postgres pg_dump -U bookdb bookdb > backup_$(date +%Y%m%d_%H%M%S).sql

# Automated backup script
#!/bin/bash
BACKUP_DIR="/backups"
DATE=$(date +%Y%m%d_%H%M%S)
docker-compose exec -T postgres pg_dump -U bookdb bookdb > $BACKUP_DIR/bookdb_$DATE.sql
find $BACKUP_DIR -name "bookdb_*.sql" -mtime +7 -delete
```

### Model Backup

```bash
# Backup trained models
tar -czf models_backup_$(date +%Y%m%d).tar.gz data/models/

# Sync to cloud storage
aws s3 sync data/models/ s3://your-bucket/models/
```

## Scaling

### Horizontal Scaling

```bash
# Scale with Docker Compose
docker-compose up -d --scale app=3

# Scale with Kubernetes
kubectl scale deployment book-recommender --replicas=5 -n book-recommender
```

### Auto-scaling (Kubernetes)

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: book-recommender-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: book-recommender
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

## Troubleshooting

### Common Issues

1. **Models not loading**
   ```bash
   # Check model files exist
   ls -la data/models/

   # Retrain models
   python scripts/train_models.py --model all
   ```

2. **Database connection issues**
   ```bash
   # Check database connectivity
   docker-compose exec app python -c "from book_recommender.core.config import get_settings; print(get_settings().database_url)"

   # Reset database
   docker-compose down -v
   docker-compose up -d postgres
   ```

3. **Cache issues**
   ```bash
   # Clear Redis cache
   docker-compose exec redis redis-cli FLUSHALL

   # Check Redis connectivity
   docker-compose exec redis redis-cli ping
   ```

### Log Analysis

```bash
# View application logs
docker-compose logs -f app

# View specific error logs
docker-compose logs app | grep ERROR

# Monitor real-time logs
tail -f logs/app.log
```

### Performance Monitoring

```bash
# Check system resources
docker stats

# Monitor API response times
curl -w "@curl-format.txt" -s -o /dev/null http://localhost:8000/health

# Database performance
docker-compose exec postgres psql -U bookdb -c "SELECT * FROM pg_stat_activity;"
```

## Maintenance

### Regular Tasks

1. **Update dependencies** (monthly)
   ```bash
   pip install -r requirements.txt --upgrade
   ```

2. **Clean old logs** (weekly)
   ```bash
   find logs/ -name "*.log" -mtime +30 -delete
   ```

3. **Update models** (as needed)
   ```bash
   python scripts/train_models.py --model all
   ```

4. **Security updates** (as available)
   ```bash
   docker-compose pull
   docker-compose up -d
   ```

This deployment guide provides comprehensive instructions for deploying the Book Recommender System in various environments from development to production.
