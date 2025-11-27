import torch
import cv2
import mediapipe as mp
import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from .utils import load_model, preprocess_landmarks

@dataclass
class DetectionResult:
    text: str
    confidence: float
    landmarks: Optional[np.ndarray] = None
    bbox: Optional[tuple] = None

class RealSignDetector:
    """
    Core detector for sign language recognition.
    Uses MediaPipe for landmark detection and a PyTorch model for classification.
    """
    def __init__(self, model_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = torch.device(device)
        self.model = load_model(model_path, self.device)
        self.model.eval()
        self.mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.sign_to_text = self._load_sign_mapping()  # Simple dict for ISL signs

    def _load_sign_mapping(self) -> Dict[str, str]:
        # Placeholder mapping; extend with real ISL dictionary
        return {
            "hello": "नमस्ते",
            "thank_you": "धन्यवाद",
            # Add more...
        }

    def predict(self, frame: np.ndarray, sequence_length: int = 30) -> DetectionResult:
        """
        Predict sign from a single frame (buffers sequence internally).
        For real-time, call sequentially to build sequence.
        """
        # Detect hands with MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            # Extract landmarks for both hands
            landmarks = []
            for hand_landmarks in results.multi_hand_landmarks:
                lm = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
                landmarks.append(lm)
            if len(landmarks) == 1:
                landmarks.append(np.zeros(21*3))  # Pad single hand
            combined_landmarks = np.concatenate(landmarks)  # 42 dims

            # Preprocess
            processed = preprocess_landmarks(combined_landmarks)

            # Model inference (dummy sequence buffer for demo; in prod, use LSTM/Transformer)
            with torch.no_grad():
                input_tensor = torch.tensor(processed, dtype=torch.float32).unsqueeze(0).to(self.device)
                output = self.model(input_tensor)
                pred_class = torch.argmax(output, dim=1).item()
                confidence = torch.softmax(output, dim=1).max().item()

            sign_label = f"class_{pred_class}"  # Map to actual label
            text = self.sign_to_text.get(sign_label, "Unknown")
            return DetectionResult(text=text, confidence=confidence, landmarks=combined_landmarks)
        
        return DetectionResult(text="", confidence=0.0)

    def close(self):
        self.mp_hands.close()