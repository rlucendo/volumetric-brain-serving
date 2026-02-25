"""
Main FastAPI application factory and lifespan manager.
Bootstraps the Inference Engine and attaches routers.
"""
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from structlog import get_logger

from src.api.routes import router as api_router
from src.services.inference_engine import InferenceEngine
from src.services.medical_transforms import MedicalDataProcessor

logger = get_logger("neuroseg_api.main")

# Default path where the Docker build will place the downloaded W&B model
DEFAULT_MODEL_PATH = Path("models/last.ckpt")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the startup and shutdown lifecycle of the application.
    Loads the heavy ML models into memory once during startup.
    """
    logger.info("Starting up FastAPI application lifespan")
    
    # 1. Resolve model path (can be overridden by environment variables)
    model_path = Path(os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH))
    
    # 2. Instantiate the Domain Services
    try:
        engine = InferenceEngine(model_path=model_path)
        processor = MedicalDataProcessor()
        
        # 3. Store instances in the application state for dependency injection
        app.state.engine = engine
        app.state.processor = processor
        
        logger.info("Application state fully initialized. Ready for traffic.")
        
    except Exception as e:
        logger.error(
            "CRITICAL FAILURE during startup. Could not initialize ML engines.",
            error=str(e)
        )
        raise RuntimeError("Startup failed due to ML engine initialization error.") from e

    yield # --- The application runs and serves requests here ---

    logger.info("Shutting down FastAPI application. Cleaning up resources.")
    # Free up GPU/CPU memory
    app.state.engine = None
    app.state.processor = None


def create_app() -> FastAPI:
    """Factory function to configure and return the FastAPI instance."""
    app = FastAPI(
        title="NeuroSeg-3D Serving API",
        description="Production API for Volumetric Brain Tumor Segmentation (BraTS).",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Register the endpoints
    app.include_router(api_router, prefix="/api/v1")

    return app

# The instance picked up by Uvicorn (our ASGI server)
app = create_app()