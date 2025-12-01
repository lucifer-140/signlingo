# SignLingo - Real-time Sign Language Detection

**SignLingo** is a Python-based application designed to detect and translate sign language in real-time using video input from a webcam. It aims to bridge the communication gap for the deaf community or anyone interested in learning sign language by providing immediate visual feedback.

![SignLingo Demo](screenshot_mockup.png)

## Features

*   **Real-time Detection**: Instantly translates sign language gestures into text.
*   **Deep Learning Model**: Utilizes a custom-trained Bidirectional LSTM model for accurate sequence classification.
*   **Holistic Tracking**: Integrates MediaPipe Holistic to track face, pose, and hand landmarks simultaneously.
*   **Visual Feedback**: Displays tracking lines, bounding boxes, and confidence scores on the HUD.

## Prerequisites

> [!IMPORTANT]
> **Python 3.12 is strictly required** for this project due to specific dependency versions (e.g., `torch`, `mediapipe`, `numpy`) listed in `requirements.txt`. Using other Python versions may lead to compatibility issues.

*   **Python**: Version 3.12
*   **Webcam**: Required for real-time inference.

## Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/yourusername/signlingo.git
    cd signlingo
    ```

2.  **Create a Virtual Environment**
    It is highly recommended to use a virtual environment to manage dependencies.
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # macOS/Linux
    source .venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To start the real-time detection application:

```bash
python scripts/infer_webcam.py
```

### Controls
*   `ESC`: Quit the application.
*   `t`: Toggle tracking lines (skeleton visualization).
*   `b`: Toggle bounding boxes.
*   `[` / `]`: Decrease / Increase the confidence threshold.

## How It Works (Technical Explanation)

The system operates on a pipeline that converts raw video frames into meaningful text predictions. Here is a detailed breakdown of the core components:

### 1. Feature Extraction (`scripts/extract_keypoints.py`)
The application does not process raw pixels directly for classification, as this would be computationally expensive and prone to noise. Instead, it extracts structured geometric features:

*   **MediaPipe Holistic**: This library is used to detect landmarks on the user's body.
*   **Vectorization**: For every frame, we extract specific keypoints:
    *   **Pose**: 9 selected joints (shoulders, elbows, wrists, etc.) $\times$ 3 values (x, y, visibility) = 27 dimensions.
    *   **Left Hand**: 21 joints $\times$ 2 values (x, y) = 42 dimensions.
    *   **Right Hand**: 21 joints $\times$ 2 values (x, y) = 42 dimensions.
*   **Total Feature Vector**: These are concatenated into a single **111-dimensional vector** per frame.
*   **Sequence Processing**: The model expects a fixed sequence length. We use a sliding window approach (or padding/trimming during training) to create sequences of **32 frames**.

### 2. Model Architecture (`scripts/train_mini_lstm.py`)
The core of the recognition system is a **TinyLSTM** neural network designed for efficiency:

*   **Input Layer**: Accepts sequences of shape `(Batch, 32, 111)`.
*   **LSTM Layer**: A **Bidirectional LSTM** (Long Short-Term Memory) with 64 hidden units. Bidirectional processing allows the model to understand the context of a gesture from both past and future frames within the window.
*   **Regularization**: A dropout rate of 0.1 is applied to prevent overfitting.
*   **Classification Head**:
    *   **LayerNorm**: Normalizes the output of the LSTM to stabilize training.
    *   **Linear Layer**: Maps the hidden states to the number of target classes (vocabulary size).
*   **Training**: The model is trained using the **AdamW** optimizer and **CrossEntropyLoss**, with label smoothing to improve generalization.

### 3. Real-time Inference (`scripts/infer_webcam.py`)
The inference script connects the webcam to the model:

1.  **Capture**: Reads a frame from the webcam.
2.  **Process**: Passes the frame through MediaPipe to get landmarks.
3.  **Extract**: Converts landmarks into the 111-dim feature vector.
4.  **Buffer**: Appends the vector to a rolling buffer of the last 32 frames.
5.  **Predict**: Once the buffer is full, it is passed to the `TinyLSTM` model.
6.  **Smooth**: Prediction probabilities are smoothed over time (Exponential Moving Average) to reduce flickering.
7.  **Display**: If the confidence score exceeds the threshold (default 0.60), the predicted word is displayed on the screen.

## Project Structure

*   `scripts/`
    *   `extract_keypoints.py`: Preprocesses video data into keypoint sequences for training.
    *   `train_mini_lstm.py`: Defines and trains the LSTM model.
    *   `infer_webcam.py`: Main script for real-time detection.
    *   `record_clips.py`: Helper tool to record training data.
*   `models/`: Stores trained model checkpoints (`.pth`).
*   `data_raw/`: Directory for raw video samples (if training your own).
*   `data_proc/`: Directory for processed `.npz` datasets.
*   `requirements.txt`: List of Python dependencies.

## Future Roadmap

*   [ ] Expand vocabulary with more sign language gestures.
*   [ ] Develop a user-friendly GUI (Graphical User Interface).
*   [ ] Implement Text-to-Speech integration.
*   [ ] Optimize for mobile devices.

