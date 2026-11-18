# Face, Expression, and Hand Gesture Detection

A webcam demo that uses OpenCV and MediaPipe to detect faces, estimate facial
expressions, and recognize a few basic hand gestures in real time.

## Requirements

- Python 3
- A webcam
- An internet connection on the first run to download the MediaPipe models

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
python "Face Detection.py"
```

Press `q` in the video window to quit. The script downloads the face and hand
landmarker models into a `models` folder beside the script the first time it
runs. Those model files are not included in this repository.

The face counter uses OpenCV's Haar cascade. Expressions and gestures are
model-based estimates and may vary with lighting, camera angle, and distance.
