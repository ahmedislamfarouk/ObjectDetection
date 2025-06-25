import sys
import os

from collections import deque
emotion_history = deque(maxlen=10)  # Store last 10 emotions.


import bz2
import cv2
import dlib
import time
import numpy as np
from deepface import DeepFace
from scipy.spatial import distance as dist
from imutils import face_utils

# =========== Load models ===========
detector = dlib.get_frontal_face_detector()
import os
import urllib.request

# Check and download landmark file if not present
model_path = "shape_predictor_68_face_landmarks.dat"
compressed_model = model_path + ".bz2"
if not os.path.exists(model_path):
    print("Downloading shape_predictor_68_face_landmarks.dat...")
    url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    urllib.request.urlretrieve(url, compressed_model)

    print("Extracting...")
    with bz2.BZ2File(compressed_model) as fr, open(model_path, "wb") as fw:
        fw.write(fr.read())
    os.remove(compressed_model)
    print("Download and extraction complete.")

# Load the predictor
predictor = dlib.shape_predictor(model_path)



# Facial landmark indices
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# Thresholds
EAR_THRESHOLD = 0.25
EAR_CONSEC_FRAMES = 15
MAR_THRESHOLD = 0.5
NOD_THRESHOLD = 15

# Counters and memory
blink_counter = 0
yawn_counter = 0
nod_counter = 0
prev_y = None
emotion_history = deque(maxlen=10)
neutral_like_sad = 0  # smart sad fix counter

# Start webcam
cap = cv2.VideoCapture(0)
print("Driver monitoring started...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    for rect in rects:
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        face_img = frame[y:y+h, x:x+w]

        # ========== Emotion Detection for each face ==========
        try:
            result = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
            emotion = result[0]['dominant_emotion']
        except:
            emotion = "Unknown"

        # Draw bounding box and emotion label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"Emotion: {emotion}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Facial landmark detection and fatigue detection code for each face
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # EAR (eye aspect ratio)
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = (dist.euclidean(leftEye[1], leftEye[5]) + dist.euclidean(leftEye[2], leftEye[4])) / (2.0 * dist.euclidean(leftEye[0], leftEye[3]))
        rightEAR = (dist.euclidean(rightEye[1], rightEye[5]) + dist.euclidean(rightEye[2], rightEye[4])) / (2.0 * dist.euclidean(rightEye[0], rightEye[3]))
        ear = (leftEAR + rightEAR) / 2.0

        # MAR (mouth aspect ratio)
        mouth = shape[mStart:mEnd]
        A = dist.euclidean(mouth[13], mouth[19])
        B = dist.euclidean(mouth[14], mouth[18])
        C = dist.euclidean(mouth[12], mouth[16])
        mar = (A + B) / (2.0 * C)

        # Head nod detection
        nose_point = shape[33]
        if prev_y is not None and abs(prev_y - nose_point[1]) > NOD_THRESHOLD:
            nod_counter += 1
            # Draw warning near the face
            cv2.putText(frame, "⚠ Head Nodding!", (x, y + h + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        prev_y = nose_point[1]

        # Drowsiness detection
        if ear < EAR_THRESHOLD:
            blink_counter += 1
            if blink_counter >= EAR_CONSEC_FRAMES:
                # Draw warning near the face
                cv2.putText(frame, "⚠ Drowsiness Detected!", (x, y + h + 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            blink_counter = 0

        # Yawning detection
        if mar > MAR_THRESHOLD:
            yawn_counter += 1
            cv2.putText(frame, "⚠ Yawning!", (x, y + h + 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        # Emotion correction (smart sad → neutral)
        if emotion == "sad":
            if mar < 0.3 and ear > EAR_THRESHOLD:
                neutral_like_sad += 1
            else:
                neutral_like_sad = 0

            if neutral_like_sad >= 5:
                emotion = "neutral"

        # Draw landmarks (optional)
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    cv2.imshow("Driver Emotion & Fatigue Monitor", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()