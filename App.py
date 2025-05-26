import cv2
import torch
import streamlit as st
import numpy as np
import os
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="Team B Object Detection", layout="wide")
st.title("Real-Time Object Detection for Autonomous Driving")
st.subheader("Team B: Pedestrians, Vehicles, Traffic Cones, Lane Detection")

@st.cache_resource
def load_models():
    pedestrian_vehicle_model = YOLO('yolov8m.pt')  # COCO model
    cone_model = YOLO('my_model2.pt')              # Custom cone model
    lane_model = YOLO('my_model.pt')               # Custom lane model
    return pedestrian_vehicle_model, cone_model, lane_model

pedestrian_vehicle_model, cone_model, lane_model = load_models()

st.sidebar.title("Detection Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
input_source = st.sidebar.radio("Select Input Source", ["Webcam", "Upload Image", "Upload Video"])

def draw_pedestrian_vehicle_detections(image_np, results):
    for box, cls in zip(results.boxes, results.boxes.cls):
        class_id = int(cls.item())
        conf = box.conf.item()

        if class_id == 0:  # COCO: person
            label = f"Pedestrian: {conf:.2f}"
            color = (0, 255, 0)
        elif class_id == 2:  # COCO: car
            label = f"Vehicle: {conf:.2f}"
            color = (0, 128, 255)
        else:
            continue  # skip other classes

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(image_np, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image_np, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return image_np

def draw_cone_detections(image_np):
    results = cone_model(image_np, conf=0.25)[0]
    for box, cls in zip(results.boxes, results.boxes.cls):
        if int(cls.item()) != 6:  # Replace 6 with your cone class index if needed
            continue
        conf = box.conf.item()
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = f"Cone: {conf:.2f}"
        cv2.rectangle(image_np, (x1, y1), (x2, y2), (255, 165, 0), 2)
        cv2.putText(image_np, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
    return image_np

def draw_lane_detections(image_np):
    results = lane_model(image_np, conf=0.25)[0]
    for box, cls in zip(results.boxes, results.boxes.cls):
        if int(cls.item()) != 5:  # Custom: Lane class
            continue
        conf = box.conf.item()
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = f"Lane: {conf:.2f}"
        cv2.rectangle(image_np, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(image_np, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    return image_np

def process_frame(frame, conf_threshold):
    results = pedestrian_vehicle_model(frame, conf=conf_threshold)[0]
    annotated_frame = results.orig_img.copy()
    annotated_frame = draw_pedestrian_vehicle_detections(annotated_frame, results)
    annotated_frame = draw_cone_detections(annotated_frame)
    annotated_frame = draw_lane_detections(annotated_frame)
    return annotated_frame

if input_source == "Webcam":
    st.subheader("Live Detection Feed")
    run = st.checkbox('Start Detection')
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(1)

    while run:
        ret, frame = camera.read()
        if not ret:
            st.warning("Failed to access camera. Try changing the camera index.")
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output = process_frame(frame, confidence_threshold)
        FRAME_WINDOW.image(output, channels="RGB")

    camera.release()

elif input_source == "Upload Image":
    st.subheader("Image Detection")
    uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        image_np = np.array(image)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(image_np, channels="RGB")

        with col2:
            st.subheader("Detected Objects")
            result = process_frame(image_np, confidence_threshold)
            st.image(result, channels="RGB")

elif input_source == "Upload Video":
    st.subheader("Video Detection")
    uploaded_video = st.file_uploader("Upload a video...", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        temp_file = "temp_video.mp4"
        with open(temp_file, "wb") as f:
            f.write(uploaded_video.read())

        video_cap = cv2.VideoCapture(temp_file)
        stframe = st.empty()

        play_button = st.button("Play Video Detection")
        stop_button = st.button("Stop")

        if play_button and not stop_button:
            while video_cap.isOpened():
                ret, frame = video_cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = process_frame(frame_rgb, confidence_threshold)
                stframe.image(result, channels="RGB")

        video_cap.release()

st.markdown("---")
st.markdown("""
### Team B Detection Classes
- **Pedestrians**: YOLOv8 (`yolov8m.pt`, COCO class 0)
- **Vehicles**: YOLOv8 (`yolov8m.pt`, COCO class 2)
- **Traffic Cones**: Custom YOLOv8 model (`my_model2.pt`, custom class)
- **Traffic Lanes**: Custom YOLOv8 model (`my_model.pt`, class 5)
""")
