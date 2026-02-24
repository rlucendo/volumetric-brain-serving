"""
Core inference engine wrapping the 3D U-Net model.
Manages GPU/CPU allocation, safe checkpoint loading, and sliding window inference.
"""
import os
from pathlib import Path
from typing import Any

import torch
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet  # Replace with your specific custom model import if needed
from structlog import get_logger

logger = get_logger("neuroseg_api.engine")

class InferenceEngine:
    """
    Singleton-like engine to hold the loaded model in memory and perform predictions.
    Avoids reloading the heavy .ckpt file for every incoming request.
    """

    def __init__(self, model_path: Path) -> None:
        """
        Initializes the model architecture and loads the trained weights.
        
        Args:
            model_path: Path to the PyTorch Lightning .ckpt file.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Initializing Inference Engine", device=str(self.device))
        
        self.model = self._build_architecture()
        self._load_weights(model_path)
        
        # Set to evaluation mode and optimize for inference
        self.model.eval()
        self.model.to(self.device)

    def _build_architecture(self) -> torch.nn.Module:
        """
        Constructs the blank 3D U-Net architecture.
        Note: Update these parameters to match your exact training configuration.
        """
        logger.debug("Building model architecture")
        # Assuming a standard MONAI 3D U-Net based on your payload description.
        # If you have a custom LightningModule, instantiate its core PyTorch model here.
        return UNet(
            spatial_dims=3,
            in_channels=4,      # FLAIR, T1w, T1gd, T2w
            out_channels=4,     # Background, Necrosis, Edema, Active Tumor
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm="batch",
        )

    def _load_weights(self, ckpt_path: Path) -> None:
        """
        Securely loads the checkpoint. Maps PyTorch Lightning keys to MONAI native keys.
        """
        if not ckpt_path.exists():
            logger.error("Checkpoint not found", path=str(ckpt_path))
            raise FileNotFoundError(f"Model weights not found at {ckpt_path}")

        logger.info("Loading weights into memory", path=str(ckpt_path))
        try:
            checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            state_dict = checkpoint.get("state_dict", checkpoint)
            
            clean_state_dict = {}
            for k, v in state_dict.items():
                # Strip all nested prefixes injected by PyTorch Lightning or Custom Modules.
                # This resolves the "Matryoshka" nesting effect (e.g., 'model.net.model.0.conv...').new_key = k
                while new_key.startswith("net.") or new_key.startswith("model."):
                    if new_key.startswith("net."):
                        new_key = new_key[4:] # Strip 'net.' prefix
                    elif new_key.startswith("model."):
                        new_key = new_key[6:] # Strip 'model.' prefix
                
                # At this point, new_key is the raw base layer name (e.g., '0.conv.unit0...').
                # We prepend the strict single 'model.' prefix expected by MONAI's native UNet.
                final_key = f"model.{new_key}"
                clean_state_dict[final_key] = v
            
            # Use strict=True to guarantee 100% weight matching
            self.model.load_state_dict(clean_state_dict, strict=True)
            logger.info("Weights loaded successfully")
            
        except Exception as e:
            logger.error("Failed to load checkpoint", error=str(e))
            raise RuntimeError(f"Weight loading failed: {e}") from e

    @torch.no_grad()
    def predict(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Executes sliding window inference on a preprocessed 3D tensor.
        
        Args:
            input_tensor: The 4D tensor (Channels, Depth, Height, Width) from the Preprocessor.
            
        Returns:
            A 3D tensor representing the predicted class mask (0, 1, 2, 3).
        """
        logger.info("Executing sliding window inference")
        
        # Add batch dimension (B, C, D, H, W) expected by the model
        batched_input = input_tensor.unsqueeze(0).to(self.device)

        # Sliding window parameters (must match or be smaller than your training patch size)
        roi_size = (96, 96, 96) 
        sw_batch_size = 4  # Number of overlapping windows to process in parallel

        try:
            # Get logits
            logits = sliding_window_inference(
                inputs=batched_input,
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                predictor=self.model,
                overlap=0.5
            )
            
            # Apply argmax across the class channel (dim 1) to get the final segmentation mask
            predicted_mask = torch.argmax(logits, dim=1).squeeze(0) # Remove batch dim
            
            logger.info("Inference complete")
            return predicted_mask.cpu()

        except Exception as e:
            logger.error("Inference execution failed", error=str(e))
            raise RuntimeError(f"Prediction failed: {e}") from e