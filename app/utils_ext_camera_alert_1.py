from ultralytics import YOLO
import streamlit as st
import cv2
from PIL import Image
import tempfile
from config import *
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, webrtc_streamer, WebRtcMode, RTCConfiguration
import time
import torch
from torchvision.transforms import functional as F
import time
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

def load_model(model_path):
    model = torch.hub.load('ultralytics/yolov8', 'yolov8')
    model.eval()
    return model

def detect(image):
    # Load pre-trained model
    model = load_model('best.pt')

    # Preprocess image
    img = F.to_tensor(image)
    img = img.unsqueeze(0)  # add batch dimension

    # Perform inference
    with torch.no_grad():
        predictions = model(img)

    # Postprocess predictions
    boxes, scores, classes = [], [], []
    for pred in predictions:
        boxes.append(pred['boxes'].tolist())
        scores.append(pred['scores'].tolist())
        classes.append(pred['labels'].tolist())

    return image, (boxes, scores, classes)


def plot_one_box(xyxy, img, label, color, line_thickness):
    cv2.rectangle(img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, thickness=line_thickness)
    cv2.putText(img, label, (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

class DetectionTransformer(VideoTransformerBase):
    def __init__(self):
        self.last_event = None
        self.last_event_time = None
        self.names = ['falling', 'sitting', 'standing']  # Define your class names
        self.colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Define your colors


    def transform(self, frame):
        img, detections = detect(frame)  # Apply YOLOv8 detection

        for *xyxy, conf, cls in detections:
            label = f'{self.names[int(cls)]}: {conf:.2f}'
            plot_one_box(xyxy, img, label=label, color=self.colors[int(cls)], line_thickness=3)

            # Check for "falling" and "standing" events within 2 seconds
            if label in ["falling", "standing"]:
                if self.last_event == "falling" and label == "standing" and time.time() - self.last_event_time <= 2:
                    print("Real fall")
                    st.sidebar.text("Real fall detected")  # Display the message in the sidebar
                self.last_event = label
                self.last_event_time = time.time()

        return img

def play_webcam_alert():
    webrtc_streamer(key="example", video_transformer_factory=DetectionTransformer)
