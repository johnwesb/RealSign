#!/usr/bin/env python
"""
Main CLI for inference.
"""
import argparse
import cv2
import sys
from pathlib import Path
from src.pipeline import RealSignPipeline
from src.utils import setup_logging

logger = setup_logging("run")

def main():
    parser = argparse.ArgumentParser(description="Run RealSign inference")
    parser.add_argument("--mode", choices=["webcam", "video"], default="webcam", help="Input mode")
    parser.add_argument("--input", type=str, help="Video file path")
    parser.add_argument("--model", type=str, default="models/isl_latest.pth", help="Model path")
    parser.add_argument("--display", action="store_true", help="Show video with overlays")
    parser.add_argument("--speech", action="store_true", help="Enable speech output")
    parser.add_argument("--log", type=str, help="CSV log file")
    args = parser.parse_args()

    if args.mode == "video" and not args.input:
        logger.error("Video mode requires --input")
        sys.exit(1)

    pipeline = RealSignPipeline(args.model, use_speech=args.speech)

    if args.mode == "webcam":
        cap = cv2.VideoCapture(0)
        stream = (cap.read()[1] for _ in iter(int, 1))  # Infinite generator
    else:
        cap = cv2.VideoCapture(args.input)
        stream = (cap.read()[1] for _ in iter(int, 1))

    try:
        for result in pipeline.process_stream(stream, display=args.display):
            if result.text:
                logger.info(f"Detected: {result.text} (conf: {result.confidence:.2f})")
                if args.log:
                    # Append to CSV
                    pass  # Implement CSV writer
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()