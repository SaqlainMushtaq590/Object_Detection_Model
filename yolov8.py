import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
from ultralytics import YOLO
import av
import numpy as np
from PIL import Image
import tempfile

# High Accuracy Model
model = YOLO("yolov8l.pt")



def process_frame(img, conf_threshold):

    # Higher image size = better accuracy
    results = model(
        img,
        conf=conf_threshold,
        imgsz=1280,
        verbose=False
    )

    for r in results:
        for box in r.boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            conf_score = float(box.conf[0])
            cls_id = int(box.cls[0])

            label = model.names[cls_id]

            # Better object grouping
            if label == "person":
                color = (0, 255, 0)
                display_name = "Person"

            elif label in ["car", "truck", "bus", "motorcycle"]:
                color = (255, 0, 0)
                display_name = "Vehicle"

            else:
                color = (0, 0, 255)
                display_name = label

            display_text = f"{display_name} {conf_score:.0%}"

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            cv2.putText(
                img,
                display_text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

    return img


class VideoTransformer(VideoProcessorBase):

    def __init__(self):
        self.conf_threshold = 0.40

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        processed_img = process_frame(
            img,
            st.session_state.get("conf_slider", 0.40)
        )

        return av.VideoFrame.from_ndarray(
            processed_img,
            format="bgr24"
        )


# Streamlit Page Settings
st.set_page_config(
    page_title="AI Based Object Detection using YOLOv8",
    layout="wide"
)

st.title("AI Based Object Detector")


# Sidebar
st.sidebar.title("Control Panel")

app_mode = st.sidebar.selectbox(
    "Choose Input",
    ["Live Camera", "Upload Image", "Upload Video"]
)

conf_threshold = st.sidebar.slider(
    "Set Confidence Level",
    min_value=0.0,
    max_value=1.0,
    value=0.40,
    key="conf_slider"
)

st.sidebar.info(
    "A higher threshold means the AI must be more sure before showing a box."
)


# Main Panel
st.markdown(f"""
### AI Detection Panel

**Confidence Threshold:** `{conf_threshold:.0%}`

🟢 Person &nbsp;&nbsp;
🔵 Vehicle &nbsp;&nbsp;
🔴 Other Objects
""")


# LIVE CAMERA
if app_mode == "Live Camera":

    webrtc_streamer(
        key="live-detection",

        video_processor_factory=VideoTransformer,

        rtc_configuration={
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]}
            ]
        },

        media_stream_constraints={
            "video": True,
            "audio": False
        }
    )


# IMAGE UPLOAD
elif app_mode == "Upload Image":

    uploaded_file = st.file_uploader(
        "Choose an Image...",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:

        image = Image.open(uploaded_file)

        img_array = np.array(image)

        cv_img = cv2.cvtColor(
            img_array,
            cv2.COLOR_RGB2BGR
        )

        processed_img = process_frame(
            cv_img,
            conf_threshold
        )

        st.image(
            cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB),
            use_container_width=True
        )


# VIDEO UPLOAD
elif app_mode == "Upload Video":

    uploaded_video = st.file_uploader(
        "Choose a Video...",
        type=["mp4", "mov", "avi"]
    )

    if uploaded_video:

        tfile = tempfile.NamedTemporaryFile(delete=False)

        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)

        frame_placeholder = st.empty()

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            processed_img = process_frame(
                frame,
                conf_threshold
            )

            frame_placeholder.image(
                cv2.cvtColor(
                    processed_img,
                    cv2.COLOR_BGR2RGB
                ),
                use_container_width=True
            )

        cap.release()