import streamlit as st
import cv2
import numpy as np
from roboflow import Roboflow
import tempfile
import os

st.set_page_config(page_title="Mask Detection App", layout="wide")
st.title("🧠 Mask Detection using Roboflow + Webcam")

# Initialize Roboflow model
rf = Roboflow(api_key="Ng9oM0kz4EkpGm7tVsbN")
project = rf.workspace().project("facemaskdet-2kw9y")
model = project.version(1).model

# Button state in Streamlit session
if 'detection' not in st.session_state:
    st.session_state.detection = False

# Start/Stop buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("▶️ Start Detection"):
        st.session_state.detection = True
with col2:
    if st.button("⛔ Stop Detection"):
        st.session_state.detection = False

frame_placeholder = st.empty()

# Start webcam only when detection is active
if st.session_state.detection:
    cap = cv2.VideoCapture(0)
    
    while st.session_state.detection:
        ret, frame = cap.read()
        if not ret:
            st.error("Webcam not working.")
            break

        # Save frame temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            temp_filename = tmp_file.name
            cv2.imwrite(temp_filename, frame)

        # Predict
        prediction = model.predict(temp_filename, confidence=40, overlap=30).json()

        # Remove temp file
        os.remove(temp_filename)

        # Draw boxes
        for pred in prediction["predictions"]:
            x1 = int(pred["x"] - pred["width"] / 2)
            y1 = int(pred["y"] - pred["height"] / 2)
            x2 = int(pred["x"] + pred["width"] / 2)
            y2 = int(pred["y"] + pred["height"] / 2)

            label = pred["class"]
            conf = pred["confidence"]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Convert BGR to RGB for Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB")

        # Break if Stop is clicked
        if not st.session_state.detection:
            break
