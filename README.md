# RealSign: Sign Language Recognition System

**RealSign** is a computer vision application designed to interpret sign language gestures into text. Utilizing the YOLOv11 architecture, the system provides real-time inference capabilities suitable for accessibility tools and communication interfaces.

## System Overview

* **Application Name:** RealSign
* **Model Architecture:** YOLOv11 (Nano)
* **Deployment Environment:** Streamlit / Hugging Face Spaces
* **Primary Function:** Optical Character Recognition for Hand Gestures (A-Z)

## Key Features

* **Real-Time Inference:** Low-latency processing of video frames for immediate feedback.
* **Dual Input Modes:** Supports both direct webcam feed and static image file uploads.
* **Confidence Metrics:** Displays probability scores for each detection to ensure reliability.
* **Responsive Interface:** Professional UI design adaptable to various display resolutions.

## Technical Architecture

The system is built upon a modular Python stack:

1.  **Core Inference Engine:** Ultralytics YOLOv11 trained on a dataset of 87,000+ annotated images.
2.  **Frontend Framework:** Streamlit for web-based rendering and state management.
3.  **Image Processing:** OpenCV and PIL for matrix manipulation and pre-processing.

## Installation and Usage

### Prerequisites

* Python 3.9 or higher
* pip package manager

### Local Deployment

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/johnwesb/RealSign.git](https://github.com/johnwesb/RealSign.git)
    cd RealSign
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Execute Application**
    ```bash
    streamlit run app.py
    ```

## Cloud Deployment

This application is configured for deployment on containerized environments such as Hugging Face Spaces.

* **System Dependencies:** Requires `libgl1` (configured in `packages.txt`).
* **Python Dependencies:** Listed in `requirements.txt`.

## License

This project is distributed under the MIT License.
