import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO

MODEL_PATH = "models/yolo11m.pt"
model = YOLO(MODEL_PATH)

# Camera Parameters (Modify based on actual camera specs)
SENSOR_HEIGHT = 4.8  # mm
FOCAL_LENGTH_MM = 24  # mm
IMAGE_HEIGHT_PIXELS = 1080

FOCAL_LENGTH_PIXELS = (FOCAL_LENGTH_MM / SENSOR_HEIGHT) * IMAGE_HEIGHT_PIXELS

KNOWN_HEIGHTS = {"car": 1.5, "bus": 3.0, "truck": 3.5, "motorbike": 1.0}  
ALERT_DISTANCE_SELF = 5  # Increased threshold for better safety
ALERT_DISTANCE_OTHER = 3  # Adjusted for realistic accident detection

st.title("🚗 Object Detection & Proximity Alert System ⚠️")

def estimate_distance(box, class_name, image_height):
    x_min, y_min, x_max, y_max = box
    box_height = max(y_max - y_min, 1)
    if class_name in KNOWN_HEIGHTS:
        real_height = KNOWN_HEIGHTS[class_name]
        distance = (FOCAL_LENGTH_PIXELS * real_height) / (box_height * (image_height / IMAGE_HEIGHT_PIXELS))
        return round(distance, 2)
    return None

def calculate_proximity(boxes, distances):
    proximity_info = []
    centroids = []
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        centroid_x = (x_min + x_max) / 2
        centroid_y = (y_min + y_max) / 2
        centroids.append((centroid_x, centroid_y))
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            pixel_distance = np.linalg.norm(np.array(centroids[i]) - np.array(centroids[j]))
            if distances[i] is not None and distances[j] is not None:
                real_distance = abs(distances[i] - distances[j])
            else:
                real_distance = pixel_distance * 0.05  # Adjusted scale factor for better accuracy
            proximity_info.append((i, j, round(real_distance, 2)))
    return proximity_info

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    image_cv = np.array(image)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
    image_height, _, _ = image_cv.shape

    results = model.predict(image_cv, conf=0.5, iou=0.4)
    result = results[0]

    boxes = result.boxes.xyxy.cpu().numpy()
    labels = result.boxes.cls.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()

    distances = [estimate_distance(box, result.names[int(label)], image_height) for box, label in zip(boxes, labels)]

    proximity_results = calculate_proximity(boxes, distances)

    st.image(result.plot(), caption="Detection Results", use_column_width=True)

    st.subheader("🚗 Detected Vehicles & Distances:")
    table_data = []
    
    for i, (box, label, conf, dist) in enumerate(zip(boxes, labels, confidences, distances)):
        dist_text = f"{dist:.2f} m" if dist else "Unknown"
        table_data.append([i, result.names[int(label)], f"{conf:.2f}", dist_text])

    df = pd.DataFrame(table_data, columns=["Object ID", "Type", "Confidence", "Distance from Camera"])
    st.table(df)

    warning_issued = False
    alert_messages = []

    for i, dist in enumerate(distances):
        if dist is not None and dist < ALERT_DISTANCE_SELF:
            alert_messages.append(f"🚨 **WARNING: STOP VEHICLE!** Object {i} is only {dist:.2f}m away!")
            warning_issued = True

    accident_table = []
    for i, j, real_distance in proximity_results:
        if real_distance is not None and real_distance < ALERT_DISTANCE_OTHER:
            accident_table.append([f"Object {i}", f"Object {j}", f"{real_distance:.2f} m"])
            alert_messages.append(f"⚠️ **ACCIDENT AHEAD!** Objects {i} & {j} are only {real_distance:.2f}m apart!")
            warning_issued = True

    st.subheader("👥 Proximity Between Vehicles:")
    proximity_table = []
    for i, j, real_distance in proximity_results:
        proximity_table.append([f"Object {i}", f"Object {j}", f"{real_distance:.2f} m"])
    
    if proximity_table:
        df_proximity = pd.DataFrame(proximity_table, columns=["Vehicle 1", "Vehicle 2", "Distance"])
        st.table(df_proximity)

    if warning_issued:
        st.subheader("⚠️ **ALERTS:**")
        for msg in alert_messages:
            st.warning(msg)
    else:
        st.error("❌ **DANGER: Possible accident detected! Take immediate action.**")
