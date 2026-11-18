import time
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_DIRECTORY = Path(__file__).resolve().parent / "models"
FACE_MODEL = MODEL_DIRECTORY / "face_landmarker.task"
HAND_MODEL = MODEL_DIRECTORY / "hand_landmarker.task"

FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


def download_model(path, url):
    if not path.exists():
        MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {path.name}...")
        urlretrieve(url, path)


def classify_hand_gesture(hand_landmarks):
    points = [(point.x, point.y) for point in hand_landmarks]

    fingers = [
        points[8][1] < points[6][1],
        points[12][1] < points[10][1],
        points[16][1] < points[14][1],
        points[20][1] < points[18][1],
    ]
    thumb_up = points[4][1] < points[3][1] and points[4][1] < points[5][1]
    thumb_extended = abs(points[4][0] - points[2][0]) > 0.05

    extended_count = sum(fingers)
    if all(fingers) and thumb_extended:
        return "Open Palm"
    if not any(fingers) and thumb_up:
        return "Thumbs Up"
    if not any(fingers):
        return "Fist"
    if fingers[0] and fingers[1] and not fingers[2] and not fingers[3]:
        return "Peace"
    if fingers[0] and not any(fingers[1:]):
        return "Pointing"
    if fingers[0] and fingers[1] and fingers[2] and not fingers[3]:
        return "Three"
    if thumb_extended and thumb_up:
        return "Thumbs Up"
    return f"{extended_count + 1} Fingers"


def get_expression(face_blendshapes):
    if not face_blendshapes:
        return "Neutral"

    scores = {
        blend.category_name: blend.score
        for blend in face_blendshapes[0]
    }

    if scores.get("mouthSmile_L", 0) > 0.45 or scores.get("mouthSmile_R", 0) > 0.45:
        return "Smiling"
    if scores.get("jawOpen", 0) > 0.55:
        return "Surprised"
    if scores.get("eyeBlinkLeft", 0) > 0.7 and scores.get("eyeBlinkRight", 0) > 0.7:
        return "Eyes Closed"
    if scores.get("browDownLeft", 0) > 0.45 or scores.get("browDownRight", 0) > 0.45:
        return "Angry"
    if scores.get("browInnerUp", 0) > 0.45:
        return "Surprised"
    return "Neutral"


download_model(FACE_MODEL, FACE_MODEL_URL)
download_model(HAND_MODEL, HAND_MODEL_URL)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

if face_cascade.empty():
    print("Error: Could not load the face Haar cascade.")
    raise SystemExit(1)

base_options_face = python.BaseOptions(model_asset_path=str(FACE_MODEL))
face_options = vision.FaceLandmarkerOptions(
    base_options=base_options_face,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=5,
    output_face_blendshapes=True,
)
face_landmarker = vision.FaceLandmarker.create_from_options(face_options)

base_options_hand = python.BaseOptions(model_asset_path=str(HAND_MODEL))
hand_options = vision.HandLandmarkerOptions(
    base_options=base_options_hand,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=4,
)
hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    raise SystemExit(1)

print("Starting live face, expression, and gesture detection. Press 'q' to exit.")
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image frame.")
        break

    frame = cv2.flip(frame, 1)
    height, width = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    timestamp_ms = int((time.time() - start_time) * 1000)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )

    face_result = face_landmarker.detect_for_video(mp_image, timestamp_ms)
    expression = get_expression(face_result.face_blendshapes)

    cv2.putText(
        frame,
        f"Faces: {len(faces)}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 0),
        2,
    )

    for index, (x, y, w, h) in enumerate(faces):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        roi_gray = gray[y:y + h, x:x + w]
        label = expression if index == 0 and len(faces) == 1 else "Face"

        cv2.putText(
            frame,
            f"{label}",
            (x, max(25, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        if not eye_cascade.empty():
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 4)
            for ex, ey, ew, eh in eyes:
                cv2.rectangle(
                    frame,
                    (x + ex, y + ey),
                    (x + ex + ew, y + ey + eh),
                    (255, 0, 0),
                    1,
                )

    hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
    if hand_result.hand_landmarks:
        for hand_index, hand_landmarks in enumerate(hand_result.hand_landmarks):
            gesture = classify_hand_gesture(hand_landmarks)

            for point in hand_landmarks:
                px = int(point.x * width)
                py = int(point.y * height)
                cv2.circle(frame, (px, py), 4, (0, 165, 255), -1)

            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),
                (0, 5), (5, 6), (6, 7), (7, 8),
                (5, 9), (9, 10), (10, 11), (11, 12),
                (9, 13), (13, 14), (14, 15), (15, 16),
                (13, 17), (17, 18), (18, 19), (19, 20),
                (0, 17),
            ]
            for start, end in connections:
                start_x = int(hand_landmarks[start].x * width)
                start_y = int(hand_landmarks[start].y * height)
                end_x = int(hand_landmarks[end].x * width)
                end_y = int(hand_landmarks[end].y * height)
                cv2.line(frame, (start_x, start_y), (end_x, end_y),
                         (0, 165, 255), 2)

            wrist = hand_landmarks[0]
            label_x = int(wrist.x * width)
            label_y = max(25, int(wrist.y * height) - 20)
            cv2.putText(
                frame,
                gesture,
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 165, 255),
                2,
            )

    cv2.imshow("Face, Expression, and Gesture Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

face_landmarker.close()
hand_landmarker.close()
cap.release()
cv2.destroyAllWindows()
