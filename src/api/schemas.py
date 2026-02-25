"""
Pydantic models for API request validation and response serialization.
"""
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Schema for the health check endpoint response."""
    status: str = Field(..., description="Current status of the API")
    model_loaded: bool = Field(
        ...,
        description="Indicates if the inference engine is ready in memory"
    )
    device: str = Field(..., description="Hardware device currently used for inference (cpu/cuda)")

class PredictionMetrics(BaseModel):
    """Optional schema if we want to return JSON metrics (e.g., tumor volume) instead of a file."""
    tumor_volume_mm3: float = Field(
        ...,
        description="Total volume of the predicted tumor in cubic millimeters"
    )
    processing_time_sec: float = Field(..., description="Time taken to run the inference pipeline")