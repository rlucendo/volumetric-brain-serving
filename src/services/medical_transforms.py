"""
Medical imaging transformation pipeline using MONAI.
Isolates spatial and intensity normalizations for 3D Brain MRI.
"""
from typing import Any

from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    Spacingd,
)
from structlog import get_logger

logger = get_logger("neuroseg_api.transforms")

class MedicalDataProcessor:
    """
    Handles the ETL process for medical volumetric data.
    Ensures input NIfTI files meet the strict tensor shapes and spatial 
    properties required by the NeuroSeg-3D model.
    """

    def __init__(self) -> None:
        """Initializes the deterministic MONAI transform pipeline for inference."""
        self.keys = ["image"]
        logger.info("Initializing MONAI inference transform pipeline")
        
        # Define the strict clinical transformations based on the training configuration
        self.transform_pipeline = Compose(
            [
                LoadImaged(keys=self.keys),
                EnsureChannelFirstd(keys=self.keys),
                # Standardize to RAS (Right, Anterior, Superior) neurological orientation
                Orientationd(keys=self.keys, axcodes="RAS"),
                # Resample to isotropic voxel spacing (adjust to your model's training config)
                Spacingd(
                    keys=self.keys,
                    pixdim=(1.0, 1.0, 1.0),
                    mode=("bilinear"),
                ),
                # Z-score normalization (zero mean, unit variance)
                NormalizeIntensityd(keys=self.keys, nonzero=True, channel_wise=True),
            ]
        )

    def preprocess(self, file_path: str) -> dict[str, Any]:
        """
        Applies the transformation pipeline to a raw NIfTI file.

        Args:
            file_path: Absolute path to the .nii.gz input file.

        Returns:
            A dictionary containing the preprocessed tensor and its original metadata.
            
        Raises:
            RuntimeError: If MONAI fails to load or process the volume.
        """
        logger.info("Starting preprocessing", file_path=file_path)
        data_dict = {"image": file_path}

        try:
            processed_data = self.transform_pipeline(data_dict)
            
            # Extract spatial shape to log memory footprint
            tensor_shape = processed_data["image"].shape
            logger.info("Preprocessing complete", tensor_shape=str(tensor_shape))
            
            return processed_data
            
        except Exception as e:
            logger.error("Failed to preprocess medical volume", error=str(e), file_path=file_path)
            raise RuntimeError(f"Preprocessing pipeline failed: {e}") from e