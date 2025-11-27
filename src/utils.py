import torch
import yaml
import logging
from pathlib import Path
import numpy as np

def load_model(model_path: str, device: torch.device) -> torch.nn.Module:
    """Load PyTorch model."""
    model = torch.nn.LSTM(input_size=42, hidden_size=128, num_layers=2, batch_first=True).to(device)  # Placeholder LSTM
    if Path(model_path).exists():
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    return model

def preprocess_landmarks(landmarks: np.ndarray, sequence_length: int = 1) -> np.ndarray:
    """Normalize landmarks."""
    # Normalize to [0,1] and flatten
    landmarks = landmarks.reshape(-1, 42)  # Assuming batched
    landmarks[:, :42:3] -= landmarks[:, 0:42:3].min()  # Center
    landmarks /= landmarks.max() + 1e-6
    return landmarks.astype(np.float32)

def setup_logging(log_dir: str):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)