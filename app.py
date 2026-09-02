import os
import pickle
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ---------------------------------------------------------
# Configuration & Constants
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "transfer_window_model.pkl"

DEFAULT_FEATURE_NAMES = [
    "yellow_cards",
    "red_cards",
    "goals",
    "assists",
    "minutes_played",
]

# Global model holder
model_state: Dict[str, Any] = {
    "model": None,
    "feature_names": DEFAULT_FEATURE_NAMES,
    "is_loaded": False,
    "error": None,
}


# ---------------------------------------------------------
# Lifespan Context Manager (Startup & Shutdown)
# ---------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load the serialized model
    if MODEL_PATH.exists():
        try:
            with open(MODEL_PATH, "rb") as f:
                loaded_model = pickle.load(f)
            model_state["model"] = loaded_model
            
            # Extract expected feature names from model if stored
            if hasattr(loaded_model, "feature_names_in_"):
                model_state["feature_names"] = list(loaded_model.feature_names_in_)
            else:
                model_state["feature_names"] = DEFAULT_FEATURE_NAMES
                
            model_state["is_loaded"] = True
            model_state["error"] = None
            print(f"[INFO] Model successfully loaded from {MODEL_PATH}")
            print(f"[INFO] Model features: {model_state['feature_names']}")
        except Exception as exc:
            model_state["is_loaded"] = False
            model_state["error"] = str(exc)
            print(f"[ERROR] Failed to load model: {exc}")
    else:
        model_state["is_loaded"] = False
        model_state["error"] = f"Model file not found at {MODEL_PATH}"
        print(f"[WARNING] Model file not found at {MODEL_PATH}")

    yield

    # Shutdown
    model_state["model"] = None
    model_state["is_loaded"] = False


# ---------------------------------------------------------
# FastAPI App Initialization
# ---------------------------------------------------------
app = FastAPI(
    title="Transfer Window Market Value Prediction API",
    description=(
        "FastAPI service to predict football player market valuations (in EUR) "
        "based strictly on on-pitch performance statistics (goals, assists, cards, minutes played)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Enable Cross-Origin Resource Sharing (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------
class PlayerStatsInput(BaseModel):
    goals: int = Field(
        default=0,
        ge=0,
        description="Number of goals scored",
        examples=[2],
    )
    assists: int = Field(
        default=0,
        ge=0,
        description="Number of assists provided",
        examples=[1],
    )
    minutes_played: int = Field(
        default=90,
        ge=0,
        description="Total minutes played",
        examples=[90],
    )
    yellow_cards: int = Field(
        default=0,
        ge=0,
        description="Number of yellow cards received",
        examples=[0],
    )
    red_cards: int = Field(
        default=0,
        ge=0,
        description="Number of red cards received",
        examples=[0],
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "goals": 2,
                "assists": 1,
                "minutes_played": 90,
                "yellow_cards": 0,
                "red_cards": 0,
            }
        }
    }


class BatchPlayerStatsInput(BaseModel):
    players: List[PlayerStatsInput] = Field(
        ...,
        description="List of player match records for bulk valuation prediction",
    )


class SinglePredictionResponse(BaseModel):
    predicted_market_value_eur: float = Field(
        ...,
        description="Estimated market value in EUR",
    )
    formatted_market_value: str = Field(
        ...,
        description="Formatted currency string (e.g., €12,500,000.00)",
    )
    input_features: Dict[str, Any] = Field(
        ...,
        description="Echo of the input features used for the prediction",
    )


class BatchPredictionResponse(BaseModel):
    total_records: int
    predictions: List[SinglePredictionResponse]


class ModelInfoResponse(BaseModel):
    status: str
    is_loaded: bool
    model_type: Optional[str] = None
    expected_features: List[str]
    coefficients: Optional[Dict[str, float]] = None
    intercept: Optional[float] = None
    error: Optional[str] = None


# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def format_currency(amount: float) -> str:
    """Format float value into EUR currency representation."""
    if amount < 0:
        return f"-€{abs(amount):,.2f}"
    return f"€{amount:,.2f}"


def get_model():
    """Retrieve the loaded model or raise HTTP 503 error."""
    if not model_state["is_loaded"] or model_state["model"] is None:
        error_detail = model_state.get("error") or "Model is not loaded."
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model unavailable: {error_detail}",
        )
    return model_state["model"]


def run_inference(features_df: pd.DataFrame) -> List[float]:
    """Execute prediction using the loaded regression model."""
    model = get_model()
    try:
        features_order = model_state["feature_names"]
        # Ensure correct column ordering matching training data
        aligned_df = features_df[features_order]
        predictions = model.predict(aligned_df)
        return [float(p) for p in predictions]
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {exc}",
        )


# ---------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------
@app.get("/", tags=["General"])
async def root():
    """Root endpoint providing service metadata and helpful links."""
    return {
        "service": "Transfer Window Market Value Prediction API",
        "version": "1.0.0",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "status": "ready" if model_state["is_loaded"] else "model_not_ready",
        "features": model_state["feature_names"],
        "endpoints": {
            "predict_single": "POST /predict",
            "predict_batch": "POST /predict/batch",
            "health": "GET /health",
            "model_info": "GET /model-info",
        },
    }


@app.get("/health", tags=["Monitoring"])
async def health_check():
    """Health check endpoint for monitoring tools."""
    return {
        "status": "healthy" if model_state["is_loaded"] else "degraded",
        "model_loaded": model_state["is_loaded"],
        "features": model_state["feature_names"],
        "error": model_state["error"],
    }


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model Info"])
async def model_info():
    """Retrieve details about the trained model, features, and coefficients."""
    model = model_state["model"]
    feature_names = model_state["feature_names"]
    
    if model is None:
        return ModelInfoResponse(
            status="error",
            is_loaded=False,
            expected_features=feature_names,
            error=model_state["error"] or "Model not loaded",
        )

    coef_dict = None
    if hasattr(model, "coef_"):
        coef_dict = {
            name: round(float(coef), 4)
            for name, coef in zip(feature_names, model.coef_)
        }

    return ModelInfoResponse(
        status="ok",
        is_loaded=True,
        model_type=type(model).__name__,
        expected_features=feature_names,
        coefficients=coef_dict,
        intercept=round(float(model.intercept_), 4) if hasattr(model, "intercept_") else None,
        error=None,
    )


@app.post(
    "/predict",
    response_model=SinglePredictionResponse,
    tags=["Prediction"],
    summary="Predict player market value from match statistics",
)
async def predict_single(input_data: PlayerStatsInput):
    """
    Accepts player performance metrics (goals, assists, minutes, cards)
    and returns the predicted market valuation in EUR.
    """
    input_dict = input_data.model_dump()
    input_df = pd.DataFrame([input_dict])

    raw_preds = run_inference(input_df)
    predicted_val = raw_preds[0]

    return SinglePredictionResponse(
        predicted_market_value_eur=round(predicted_val, 2),
        formatted_market_value=format_currency(predicted_val),
        input_features=input_dict,
    )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Prediction"],
    summary="Predict market values for a batch of players",
)
async def predict_batch(batch_input: BatchPlayerStatsInput):
    """
    Accepts multiple player match records and runs batch valuation inference.
    """
    if not batch_input.players:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The players list cannot be empty.",
        )

    records = [p.model_dump() for p in batch_input.players]
    input_df = pd.DataFrame(records)

    raw_preds = run_inference(input_df)

    results = []
    for rec, pred in zip(records, raw_preds):
        results.append(
            SinglePredictionResponse(
                predicted_market_value_eur=round(pred, 2),
                formatted_market_value=format_currency(pred),
                input_features=rec,
            )
        )

    return BatchPredictionResponse(
        total_records=len(results),
        predictions=results,
    )


# ---------------------------------------------------------
# Local Execution Entrypoint
# ---------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
