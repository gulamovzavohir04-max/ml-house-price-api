# ML House Price Prediction API
Machine Learning API for predicting California house prices using FastAPI and Scikit-learn.
## Features
- Train ML model with Linear Regression
- Save model with joblib
- FastAPI REST API
- Swagger documentation
- /predict endpoint
- Docker support
## API Documentation
Swagger UI:
https://ml-house-price-api-production.up.railway.app/docs
## Tech Stack
- Python
- Pandas
- Scikit-learn
- FastAPI
- Uvicorn
- Joblib
- Docker

## Endpoints

### GET /
Returns API status message.

### GET /health
Health check endpoint.

### POST /predict
Predict house price from input features.

Example request:

`json
{
  "MedInc": 8.3252,
  "HouseAge": 41.0,
  "AveRooms": 6.984127,
  "AveBedrms": 1.023810,
  "Population": 322.0,
  "AveOccup": 2.555556,
  "Latitude": 37.88,
  "Longitude": -122.23
}

Example response:
JSON
{
  "predicted_price": 4.151943055154298
}
## Run locally
uvicorn app.main:app --reload
## Run with Docker
docker build -t ml-house-price-api .
docker run -p 8000:8000 ml-house-price-api
## Author
Javohir Gulyamov
