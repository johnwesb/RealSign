# RealSign

RealSign — realtime sign-language detection and translation for mobile and desktop.

RealSign is an open-source project that detects sign-language gestures from camera or video input and translates them into text and speech on the go. The project currently focuses on Indian Sign Language (ISL) and is designed so contributors can add support for additional sign languages, improve models, and customize the pipeline.

## Key goals
- Realtime detection and translation of signs (ISL first)
- Lightweight models and fast inference for on-device use
- Open-source: easy for others to adapt, extend, and contribute

## Features
- Realtime webcam and mobile-friendly inference
- Translate recognized signs to text and optional speech output
- Extensible pipeline for adding new languages or gesture sets
- Modular: separate detection, recognition, and translation components
- Tools and scripts to train new models from labelled datasets

## Demo
(Replace with a GIF or screenshot in the repository)

Example: webcam -> detected sign activated -> translated text displayed and optionally spoken aloud.

## Supported languages
- Primary: Indian Sign Language (ISL)
- Planned: community-driven support for additional national and regional sign languages

## Getting started

### Prerequisites
- Python 3.8+
- A modern CPU; GPU recommended for training

Clone the repo:
```
git clone https://github.com/johnwesb/RealSign.git
cd RealSign
```
Create a virtual environment and install dependencies:
```
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate

pip install -r requirements.txt
```
Note: Update requirements.txt with the runtime dependencies (OpenCV, numpy, torch/tensorflow, mediapipe, etc.) to match your preferred framework.

## Quick usage

### Realtime webcam (example)
```
python run.py --mode webcam --model models/isl_latest.pth
```
### From video file
```
python run.py --mode video --input path/to/video.mp4 --model models/isl_latest.pth
```
### Output options
- `--display`: show bounding boxes, landmarks, and translation overlay
- `--speech`: enable synthesized speech output (requires a TTS engine)
- `--log`: write recognized sign labels and timestamps to a CSV

## Python API (example)
```
from realsign import RealSignDetector

detector = RealSignDetector(model_path="models/isl_latest.pth")
for frame in camera_stream:
    result = detector.predict(frame)
    print(result.text)  # translated text
```
## Training a model
- Prepare a labelled dataset in the expected format (see docs/dataset.md or examples/)
- Example training command:
```
python train.py --config configs/isl_train.yaml --output-dir models/
```
- After training, export your model for inference (ONNX/TFLite) for mobile deployment

## Model export & mobile deployment
- Export to ONNX for cross-framework compatibility
- Convert ONNX to TFLite or CoreML for mobile deployment
- Optimize and quantize models to reduce size and increase speed for on-device use

## Repository layout (suggested)
- /src — core python modules and pipeline code
- /models — pretrained models and exported artifacts (not checked in if large)
- /data — dataset helpers and sample data
- /notebooks — experiments and demos
- /scripts — run/training/inference convenience scripts
- /docs — documentation, dataset format, model cards

## Contributing
RealSign is open-source and welcomes contributions.
- Open an issue to discuss major changes before starting
- Fork the repo, create a branch, and make a focused PR
- Follow the coding style used in the project
- Add tests where appropriate and update documentation
- Please include model/benchmark details when contributing pretrained weights

## Roadmap (community-driven)
- Improve recognition accuracy for ISL across dialects and lighting conditions
- Add more sign languages (ASL, BSL, regional variants) via community datasets
- Provide lightweight mobile models with on-device inference examples
- Build an accessible demo app (Android/iOS) and sample UI components

## Privacy & ethics
- Obtain consent before collecting or sharing video data
- Avoid using the project to profile or surveil individuals
- Sign-language recognition systems should augment communication and not replace human interpreters in high-stakes contexts without thorough validation

## License
This project is open-source. Add a LICENSE file (MIT, Apache-2.0, GPL-3.0, or other) to the repository to declare the project license.

## Acknowledgements
- List of datasets, libraries, and contributors (OpenCV, PyTorch/TensorFlow, mediapipe, etc.)

## Contacts & support
- Repo issues: use GitHub Issues for bugs and feature requests
- Pull requests: welcome — create a PR against main
```