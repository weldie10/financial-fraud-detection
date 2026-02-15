# Explainability API

Production-ready REST API for on-demand SHAP explanations.

## Overview

The Explainability API provides a microservice interface for generating SHAP explanations for fraud detection predictions. It can be deployed independently and integrated with any frontend or application.

## Features

- **On-Demand Explanations**: Generate SHAP explanations for new predictions in real-time
- **Batch Processing**: Explain multiple transactions at once
- **Feature Importance**: Get aggregated feature importance across instances
- **Cached Explainers**: SHAP explainers are cached for performance
- **RESTful API**: Standard REST endpoints for easy integration

## Installation

```bash
pip install flask flask-cors
```

## Running the API

### Development

```bash
python api/explainability_api.py
```

The API will run on `http://localhost:5000`

### Production

```bash
# Using gunicorn (recommended)
gunicorn -w 4 -b 0.0.0.0:5000 api.explainability_api:app

# Using uWSGI
uwsgi --http :5000 --wsgi-file api/explainability_api.py --callable app
```

## API Endpoints

### Health Check

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "service_ready": true
}
```

### Initialize Service

```http
POST /initialize
Content-Type: application/json

{
  "model_path": "models/fraud_detection_model.joblib",
  "background_data_path": "data/processed/fraud_data_processed.csv"
}
```

**Response:**
```json
{
  "status": "initialized",
  "explainer_type": "tree"
}
```

### Explain Single Prediction

```http
POST /explain
Content-Type: application/json

{
  "transaction": {
    "Amount": 100.0,
    "time_since_signup": 24.0,
    "age": 35,
    "hour_of_day": 12,
    "day_of_week": 1,
    "transaction_count_1h": 1
  }
}
```

**Response:**
```json
{
  "prediction": 0,
  "fraud_probability": 0.15,
  "normal_probability": 0.85,
  "shap_values": [0.1, -0.05, 0.02, ...],
  "feature_importance": [
    {
      "feature": "time_since_signup",
      "value": 24.0,
      "shap_value": -0.15,
      "abs_shap_value": 0.15
    },
    ...
  ],
  "top_features": [...],
  "plot_data": {...}
}
```

### Explain Batch

```http
POST /explain_batch
Content-Type: application/json

{
  "transactions": [
    {"Amount": 100.0, "time_since_signup": 24.0, ...},
    {"Amount": 500.0, "time_since_signup": 0.5, ...}
  ],
  "max_instances": 100
}
```

**Response:**
```json
{
  "count": 2,
  "explanations": [
    {
      "prediction": 0,
      "fraud_probability": 0.15,
      "shap_values": [...],
      "instance_id": 0
    },
    ...
  ]
}
```

### Feature Importance Summary

```http
POST /feature_importance
Content-Type: application/json

{
  "transactions": [...],
  "top_n": 10
}
```

**Response:**
```json
{
  "feature_importance": [
    {
      "feature": "time_since_signup",
      "mean_abs_shap": 0.25
    },
    ...
  ]
}
```

## Usage Examples

### Python Client

```python
import requests

# Initialize service
response = requests.post('http://localhost:5000/initialize', json={
    'model_path': 'models/fraud_detection_model.joblib',
    'background_data_path': 'data/processed/fraud_data_processed.csv'
})

# Explain a prediction
response = requests.post('http://localhost:5000/explain', json={
    'transaction': {
        'Amount': 100.0,
        'time_since_signup': 24.0,
        'age': 35,
        'hour_of_day': 12,
        'day_of_week': 1,
        'transaction_count_1h': 1
    }
})

explanation = response.json()
print(f"Fraud Probability: {explanation['fraud_probability']:.2%}")
print(f"Top Feature: {explanation['top_features'][0]['feature']}")
```

### JavaScript/TypeScript Client

```javascript
// Initialize service
const initResponse = await fetch('http://localhost:5000/initialize', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    model_path: 'models/fraud_detection_model.joblib',
    background_data_path: 'data/processed/fraud_data_processed.csv'
  })
});

// Explain a prediction
const explainResponse = await fetch('http://localhost:5000/explain', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    transaction: {
      Amount: 100.0,
      time_since_signup: 24.0,
      age: 35,
      hour_of_day: 12,
      day_of_week: 1,
      transaction_count_1h: 1
    }
  })
});

const explanation = await explainResponse.json();
console.log(`Fraud Probability: ${explanation.fraud_probability}`);
```

### cURL

```bash
# Initialize
curl -X POST http://localhost:5000/initialize \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "models/fraud_detection_model.joblib",
    "background_data_path": "data/processed/fraud_data_processed.csv"
  }'

# Explain
curl -X POST http://localhost:5000/explain \
  -H "Content-Type: application/json" \
  -d '{
    "transaction": {
      "Amount": 100.0,
      "time_since_signup": 24.0,
      "age": 35,
      "hour_of_day": 12,
      "day_of_week": 1,
      "transaction_count_1h": 1
    }
  }'
```

## Integration with Dashboard

The dashboard can be configured to use the API instead of local service:

```python
# In dashboard/app.py
API_URL = "http://localhost:5000"

# Use API for explanations
response = requests.post(f"{API_URL}/explain", json={
    'transaction': transaction_data.to_dict('records')[0]
})
```

## Docker Deployment

```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt flask flask-cors

COPY . .

EXPOSE 5000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "api.explainability_api:app"]
```

## Performance Considerations

- **Caching**: SHAP explainers are cached after initialization
- **Batch Limits**: Batch endpoints limit to 100 instances by default
- **Background Data**: Uses sampled background data (100 rows) for efficiency
- **Tree Explainer**: Automatically uses fast TreeExplainer for tree-based models

## Security

For production deployment:
- Add authentication (API keys, OAuth)
- Use HTTPS
- Rate limiting
- Input validation
- CORS configuration

## Monitoring

Add logging and monitoring:
- Request/response logging
- Performance metrics
- Error tracking
- Health check monitoring
