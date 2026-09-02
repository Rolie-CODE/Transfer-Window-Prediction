# ⚽ Transfer Window Market Value Prediction

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning and API solution designed to estimate football (soccer) player transfer market valuations in Euros (€) based on on-pitch performance statistics and match activity.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Project Architecture & Repository Structure](#-project-architecture--repository-structure)
- [Installation & Setup](#-installation--setup)
- [Data Pipeline & Modeling](#-data-pipeline--modeling)
- [Running the API Service](#-running-the-api-service)
- [API Reference](#-api-reference)
  - [Endpoints Overview](#endpoints-overview)
  - [Predict Single Player Valuation](#1-predict-single-player-valuation)
  - [Predict Batch Player Valuations](#2-predict-batch-player-valuations)
  - [Model Information](#3-model-information)
  - [Health Check](#4-health-check)
- [Testing & Validation](#-testing--validation)
- [License](#-license)

---

## 📖 Overview

Predicting player valuations during football transfer windows is vital for clubs, scouts, and analysts. This project correlates historical player match appearances and individual in-game contributions with market valuations to produce transparent, data-driven valuations.

The system features:
1. **Automated Data Merging**: Aligns appearance events with chronological market values using as-of timestamp joins.
2. **Predictive Modeling**: Regression model quantifying the impact of goals, assists, minutes played, and disciplinary cards on market worth.
3. **High-Performance Serving**: FastAPI web service providing real-time single and batch prediction endpoints with full schema validation and interactive Swagger documentation.

---

## ✨ Key Features

- **Performance-Driven Inputs**: Operates purely on observable on-pitch performance metrics:
  - ⚽ `goals` (Goals scored)
  - 🎯 `assists` (Assists provided)
  - ⏱️ `minutes_played` (Time spent on pitch)
  - 🟨 `yellow_cards` (Disciplinary cautions)
  - 🟥 `red_cards` (Ejections)
- **FastAPI Backend**: Asynchronous, production-ready inference API with automatic request validation via Pydantic.
- **Batch Processing**: Predict valuations for full squads or match sheets in a single HTTP request.
- **Interactive Documentation**: Auto-generated Swagger UI (`/docs`) and ReDoc (`/redoc`).

---

## 📂 Project Architecture & Repository Structure

```plaintext
Transfer-Window-Prediction/
├── app.py                          # FastAPI application and inference service
├── model.ipynb                     # Jupyter notebook for EDA, training & evaluation
├── requirements.txt                # Python package dependencies
├── transfer_window_model.pkl       # Serialized trained scikit-learn model
├── scripts/
│   └── merge_market_values.py      # Data preprocessing and chronological merge script
├── data/                           # (Git-ignored) Raw and processed CSV datasets
└── README.md                       # Project documentation
```

---

## 🚀 Installation & Setup

### Prerequisites

- Python 3.10 or higher
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/Rolie-CODE/Transfer-Window-Prediction.git
cd Transfer-Window-Prediction
```

### 2. Create and Activate a Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**On macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📊 Data Pipeline & Modeling

### 1. Merging Appearances with Valuations

The script [`scripts/merge_market_values.py`](scripts/merge_market_values.py) merges player appearance records with historical valuations using backward timestamp matching:

```bash
python scripts/merge_market_values.py
```

### 2. Model Training

The regression model is trained on match-level data to evaluate market value dynamics. Run all cells in [`model.ipynb`](model.ipynb) to re-evaluate or train:

```python
from sklearn.linear_model import LinearRegression
import pickle

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Save to disk
with open('transfer_window_model.pkl', 'wb') as f:
    pickle.dump(model, f)
```

---

## 🖥️ Running the API Service

Start the FastAPI application with Uvicorn:

```bash
python app.py
```

Or directly via Uvicorn CLI with hot-reloading:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Once running, navigate to:
- **Interactive API Docs (Swagger UI)**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Alternative Docs (ReDoc)**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 📡 API Reference

### Endpoints Overview

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/predict` | Estimate market value for a single player match record |
| `POST` | `/predict/batch` | Bulk valuation for multiple player match records |
| `GET` | `/model-info` | Returns model coefficients, intercept, and feature metadata |
| `GET` | `/health` | Service health and model status check |
| `GET` | `/` | API root metadata and route directory |

---

### 1. Predict Single Player Valuation

`POST /predict`

#### Request Body
```json
{
  "goals": 2,
  "assists": 1,
  "minutes_played": 90,
  "yellow_cards": 0,
  "red_cards": 0
}
```

#### Response Body (`200 OK`)
```json
{
  "predicted_market_value_eur": 17376846.07,
  "formatted_market_value": "€17,376,846.07",
  "input_features": {
    "goals": 2,
    "assists": 1,
    "minutes_played": 90,
    "yellow_cards": 0,
    "red_cards": 0
  }
}
```

#### cURL Example
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "goals": 2,
       "assists": 1,
       "minutes_played": 90,
       "yellow_cards": 0,
       "red_cards": 0
     }'
```

---

### 2. Predict Batch Player Valuations

`POST /predict/batch`

#### Request Body
```json
{
  "players": [
    {
      "goals": 2,
      "assists": 1,
      "minutes_played": 90,
      "yellow_cards": 0,
      "red_cards": 0
    },
    {
      "goals": 0,
      "assists": 0,
      "minutes_played": 45,
      "yellow_cards": 1,
      "red_cards": 0
    }
  ]
}
```

#### Response Body (`200 OK`)
```json
{
  "total_records": 2,
  "predictions": [
    {
      "predicted_market_value_eur": 17376846.07,
      "formatted_market_value": "€17,376,846.07",
      "input_features": {
        "goals": 2,
        "assists": 1,
        "minutes_played": 90,
        "yellow_cards": 0,
        "red_cards": 0
      }
    },
    {
      "predicted_market_value_eur": 5077025.28,
      "formatted_market_value": "€5,077,025.28",
      "input_features": {
        "goals": 0,
        "assists": 0,
        "minutes_played": 45,
        "yellow_cards": 1,
        "red_cards": 0
      }
    }
  ]
}
```

---

### 3. Model Information

`GET /model-info`

#### Response Body (`200 OK`)
```json
{
  "status": "ok",
  "is_loaded": true,
  "model_type": "LinearRegression",
  "expected_features": [
    "yellow_cards",
    "red_cards",
    "goals",
    "assists",
    "minutes_played"
  ],
  "coefficients": {
    "yellow_cards": -667894.3209,
    "red_cards": -561472.638,
    "goals": 3875454.3909,
    "assists": 3028781.7995,
    "minutes_played": 18938.5976
  },
  "intercept": 4892681.7044,
  "error": null
}
```

---

### 4. Health Check

`GET /health`

#### Response Body (`200 OK`)
```json
{
  "status": "healthy",
  "model_loaded": true,
  "features": [
    "yellow_cards",
    "red_cards",
    "goals",
    "assists",
    "minutes_played"
  ],
  "error": null
}
```

---

### Python Client Snippet

```python
import requests

API_URL = "http://localhost:8000/predict"

player_stats = {
    "goals": 1,
    "assists": 2,
    "minutes_played": 85,
    "yellow_cards": 0,
    "red_cards": 0,
}

response = requests.post(API_URL, json=player_stats)

if response.status_code == 200:
    data = response.json()
    print(f"Valuation: {data['formatted_market_value']}")
else:
    print(f"Error {response.status_code}: {response.text}")
```

---

## 🧪 Testing & Validation

Run unit tests and verify endpoint responses:

```bash
python -c "
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)
with client:
    res = client.get('/health')
    assert res.status_code == 200
    print('Health check passed:', res.json())
"
```

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

