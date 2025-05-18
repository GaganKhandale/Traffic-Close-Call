import streamlit as st
import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO

MODEL_PATH = "models/yolo11m.pt"
model = YOLO(MODEL_PATH)

st.title("Traffic Object Detection in Adverse Weather Conditions")

uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "mp4"])

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    
    if uploaded_file.type in ["image/jpeg", "image/png", "image/jpg"]:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        image_cv = np.array(image)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

        results = model.predict(image_cv, conf=0.5, iou=0.5)

        st.image(results[0].plot(), caption="Detection Results", use_column_width=True)

        st.subheader("Detected Objects:")
        for box, label, conf in zip(results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.conf):
            st.write(f"Detected: {results[0].names[int(label)]}, Confidence: {conf:.2f}")

    elif uploaded_file.type == "video/mp4":
        video_path = "uploaded_video.mp4"
        with open(video_path, "wb") as f:
            f.write(file_bytes)

        results = model.predict(video_path, conf=0.5, iou=0.5, save=True)

        st.video(video_path)

        st.success(f"Video processed and saved in: {results[0].save_dir}")