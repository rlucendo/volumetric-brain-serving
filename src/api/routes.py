"""
API endpoints for system health checks and model inference.
"""
import os
import tempfile
import time
from pathlib import Path

import nibabel as nib
import numpy as np
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from structlog import get_logger

from src.api.schemas import HealthResponse

logger = get_logger("neuroseg_api.routes")
router = APIRouter()

def cleanup_temp_files(*file_paths: Path) -> None:
    """Background task to delete temporary files after the response is sent."""
    for path in file_paths:
        try:
            if path.exists():
                os.remove(path)
                logger.debug("Deleted temporary file", file_path=str(path))
        except Exception as e:
            logger.error("Failed to delete temporary file", file_path=str(path), error=str(e))


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check(request: Request) -> HealthResponse:
    """
    Kubernetes/Docker liveness and readiness probe.
    Verifies that the API is up and the model is loaded in memory.
    """
    engine = getattr(request.app.state, "engine", None)
    is_loaded = engine is not None
    
    return HealthResponse(
        status="healthy" if is_loaded else "degraded",
        model_loaded=is_loaded,
        device=str(engine.device) if is_loaded else "unknown"
    )


@router.post("/predict", tags=["Inference"])
async def predict_segmentation(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
) -> FileResponse:
    """
    Receives a multi-channel 3D NIfTI file, performs sliding window inference,
    and returns the predicted segmentation mask as a new NIfTI file.
    """
    if not file.filename.endswith(".nii.gz") and not file.filename.endswith(".nii"):
        raise HTTPException(status_code=400, detail="Only NIfTI files (.nii or .nii.gz) are supported.")

    logger.info("Received inference request", filename=file.filename)
    start_time = time.time()

    # Extract services from global state
    processor = request.app.state.processor
    engine = request.app.state.engine

    # Create temporary paths for input and output to avoid RAM saturation
    temp_dir = Path(tempfile.gettempdir())
    input_path = temp_dir / f"in_{file.filename}"
    output_path = temp_dir / f"out_mask_{file.filename}"

    try:
        # 1. Save uploaded file to disk
        with open(input_path, "wb") as buffer:
            buffer.write(await file.read())

        # 2. Preprocess (MONAI pipeline)
        processed_data = processor.preprocess(str(input_path))
        input_tensor = processed_data["image"]
        
        # 3. Model Inference (Forward pass)
        mask_tensor = engine.predict(input_tensor)

        # 4. Reconstruct the output NIfTI preserving original spatial metadata
        original_affine = processed_data["image"].meta.get("affine", np.eye(4))
        
        # Convert PyTorch tensor to Numpy array (uint8 is enough for classes 0,1,2,3)
        mask_np = mask_tensor.numpy().astype(np.uint8)
        
        output_nifti = nib.Nifti1Image(mask_np, original_affine)
        nib.save(output_nifti, str(output_path))
        
        processing_time = time.time() - start_time
        logger.info("Inference successful", processing_time_sec=round(processing_time, 2))

        # 5. Return the file and schedule cleanup
        background_tasks.add_task(cleanup_temp_files, input_path, output_path)
        
        return FileResponse(
            path=output_path,
            media_type="application/gzip",
            filename=f"segmentation_{file.filename}"
        )

    except Exception as e:
        # Ensure cleanup happens even if inference fails
        background_tasks.add_task(cleanup_temp_files, input_path)
        logger.error("Inference pipeline failed", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error during processing.") from e