from ultralytics import YOLO
import streamlit as st
import cv2
from PIL import Image
import tempfile
from config import *
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, webrtc_streamer, WebRtcMode, RTCConfiguration
import numpy as np
import av
import base64
import time


def _display_detected_frames(conf, model, st_count, st_frame, image):
    """
    Display the detected objects on a video frame using the YOLOv8 model.
    :param conf (float): Confidence threshold for object detection.
    :param model (YOLOv8): An instance of the `YOLOv8` class containing the YOLOv8 model.
    :param st_frame (Streamlit object): A Streamlit object to display the detected video.
    :param image (numpy array): A numpy array representing the video frame.
    :return: None
    """
    # Resize the image to a standard size
    #image = cv2.resize(image, (720, int(720 * (9 / 16))))

    # Predict the objects in the image using YOLOv8 model
    res = model.predict(image, conf=conf)

    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )


# # @st.cache_resource
# def load_model(model_path):
#     """
#     Loads a YOLO object detection model from the specified model_path.

#     Parameters:
#         model_path (str): The path to the YOLO model file.

#     Returns:
#         A YOLO object detection model.
#     """
#     model = YOLO(model_path)
#     return model


def infer_uploaded_image(conf, model):
    """
    Execute inference for uploaded image
    :param conf: Confidence of YOLOv8 model
    :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    :return: None
    """
    source_img = st.sidebar.file_uploader(
        label="Choose an image...",
        type=("jpg", "jpeg", "png", 'bmp', 'webp')
    )

    col1, col2 = st.columns(2)

    with col1:
        if source_img:
            uploaded_image = Image.open(source_img)
            # adding the uploaded image to the page with caption
            st.image(
                image=source_img,
                caption="Uploaded Image",
                use_column_width=True
            )

    if source_img:
        if st.button("Execution"):
            with st.spinner("Running..."):
                res = model.predict(uploaded_image,
                                    conf=conf)
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]

                with col2:
                    st.image(res_plotted,
                             caption="Detected Image",
                             use_column_width=True)
                    try:
                        with st.expander("Detection Results"):
                            for box in boxes:
                                st.write(box.xywh)
                    except Exception as ex:
                        st.write("No image is uploaded yet!")
                        st.write(ex)


def infer_uploaded_video(conf, model):
    """
    Execute inference for uploaded video
    :param conf: Confidence of YOLOv8 model
    :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    :return: None
    """
    source_video = st.sidebar.file_uploader(
        label="Choose a video..."
    )

    if source_video:
        st.video(source_video)

    if source_video:
        if st.button("Execution"):
            with st.spinner("Running..."):
                try:
                    OBJECT_COUNTER1 = None
                    OBJECT_COUNTER = None
                    tfile = tempfile.NamedTemporaryFile()
                    tfile.write(source_video.read())
                    vid_cap = cv2.VideoCapture(
                        tfile.name)
                    st_count = st.empty()
                    st_frame = st.empty()
                    while (vid_cap.isOpened()):
                        success, image = vid_cap.read()
                        if success:
                            _display_detected_frames(conf,
                                                     model,
                                                     st_count,
                                                     st_frame,
                                                     image
                                                     )
                        else:
                            vid_cap.release()
                            break
                except Exception as e:
                    st.error(f"Error loading video: {e}")




def process_frame(frame):   #TONY asked ChatGPT to lower resolution of input video
    # Resize the frame to lower the resolution
    frame = cv2.resize(frame, (360, 200))  # Example: Resize to 640x480
    # Further processing can be done here (e.g., adjusting encoding, further reducing quality)
    return frame


def infer_uploaded_webcam(conf, model):      # Stream with local camera
    """
    Execute inference for webcam (Plays a webcam stream on local).
    :param conf: Confidence of YOLOv8 model
    :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    :return: None
    """
    try:
        flag = st.button(
            label="Stop running"
        )
        vid_cap = cv2.VideoCapture(1)  # local camera
        st_count = st.empty()
        st_frame = st.empty()

        while not flag:
            while vid_cap.isOpened():
                success, image = vid_cap.read()
                # image = process_frame(image)  #Tony asked ChatGPT
                # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  #Tony asked ChatGPT
                if success:
                    _display_detected_frames(
                        conf,
                        model,
                        st_count,
                        st_frame,
                        image
                    )
                else:
                    vid_cap.release()
                    break
    except Exception as e:
        st.error(f"Error loading video: {str(e)}")

def play_webcam(conf, model):   # Stream on cloud (global)
    """
    Plays a webcam stream on cloud. Detects Objects in real-time using the YOLO object detection model.

    Returns:
        None

    Raises:
        None
    """
    # st.sidebar.title("Webcam Object Detection")

    def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")


        orig_h, orig_w = image.shape[0:2]
        width = 720  # Set the desired width for processing

        # cv2.resize used in a forked thread may cause memory leaks
        processed_image = np.asarray(Image.fromarray(image).resize((width, int(width * orig_h / orig_w))))

        if model is not None:
            # Perform object detection using YOLO model
            res = model.predict(processed_image, conf=conf)
            # print(f'resboxes: {res.boxes}')

            # Plot the detected objects on the video frame
            res_plotted = res[0].plot()
            # print(f'resplotted: {res_plotted}')


        return av.VideoFrame.from_ndarray(res_plotted, format="bgr24")


    webrtc_streamer(
        key="example",
        # video_transformer_factory=lambda: MyVideoTransformer(conf, model),
        video_frame_callback = video_frame_callback,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
    )
