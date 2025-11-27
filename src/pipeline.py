import cv2
from typing import Callable, Iterator
from .detector import RealSignDetector, DetectionResult
from .translator import Translator

class RealSignPipeline:
    """
    Orchestrates the full pipeline: input -> detection -> translation -> output.
    """
    def __init__(self, model_path: str, use_speech: bool = False):
        self.detector = RealSignDetector(model_path)
        self.translator = Translator(use_speech=use_speech)
        self.sequence_buffer = []  # For temporal smoothing (extend as needed)

    def process_stream(self, stream: Iterator[np.ndarray], display: bool = True) -> Iterator[DetectionResult]:
        """
        Process video stream (webcam or file).
        Yields translated results.
        """
        for frame in stream:
            result = self.detector.predict(frame)
            if result.confidence > 0.5:  # Threshold
                self.sequence_buffer.append(result)
                if len(self.sequence_buffer) > 10:  # Sliding window
                    smoothed_result = self._smooth_sequence(self.sequence_buffer[-10:])
                    translated = self.translator.translate(smoothed_result)
                    yield translated
            
            if display:
                self._display_frame(frame, result)
        
        self.detector.close()

    def _smooth_sequence(self, results: list) -> DetectionResult:
        # Simple majority vote; use HMM or better for prod
        texts = [r.text for r in results if r.text]
        if texts:
            from collections import Counter
            most_common = Counter(texts).most_common(1)[0][0]
            return DetectionResult(text=most_common, confidence=0.8)  # Avg conf
        return DetectionResult(text="", confidence=0.0)

    def _display_frame(self, frame: np.ndarray, result: DetectionResult):
        # Overlay text and bbox (simplified)
        cv2.putText(frame, result.text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("RealSign", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()