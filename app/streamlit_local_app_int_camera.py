from pathlib import Path
import streamlit as st
from ultralytics import YOLO
from utils_int_camera import infer_uploaded_image, infer_uploaded_video, infer_uploaded_webcam, play_webcam
from PIL import Image

# setting page layout
# st.set_page_config(
#     page_title="Interactive Interface for YOLOv8",
#     page_icon="🤖",
#     layout="wide",
#     initial_sidebar_state="expanded"
#     )

st.set_page_config(
    page_title="Fall Detector",
    # page_icon=":abc:",
    # layout="wide",

    initial_sidebar_state="expanded"
)

# Load the image for the title
title_image = Image.open("camera_icon.png")

# Display the title image
st.image(title_image, use_column_width=True)

# sidebar
st.sidebar.header("Model Config")
# model = YOLO('yolov8s.pt')
model = YOLO('best.pt')

# image/video options
st.sidebar.header("Input Config")
source_selectbox = st.sidebar.selectbox(
    "Select Source",
    ["Webcam", "Image", "Video"]
)

# confidence = float(st.sidebar.slider(
#     "Select Model Confidence", 30, 100, 50)) / 100

source_img = None
if source_selectbox == "Video": # Video
    infer_uploaded_video(conf=0.5, model=model)
elif source_selectbox == "Image": # Image
    infer_uploaded_image(conf=0.5, model=model)
elif source_selectbox == "Webcam": # Webcam
    # infer_uploaded_webcam(confidence, model)
    infer_uploaded_webcam(conf=0.5, model=model)
else:
    st.error("Currently only 'Image' and 'Video' source are implemented")
