import streamlit as st
import cv2
import av
import time
import numpy as np
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# ==========================================
# 1. PAGE CONFIG
# ==========================================
st.set_page_config(page_title="RealSign Live", layout="wide", page_icon="📹")

# ==========================================
# 2. SESSION STATE (Memory)
# ==========================================
if 'sentence' not in st.session_state:
    st.session_state['sentence'] = ""
if 'last_pred' not in st.session_state:
    st.session_state['last_pred'] = None
if 'pred_time' not in st.session_state:
    st.session_state['pred_time'] = 0

# ==========================================
# 3. LOAD MODEL
# ==========================================
try:
    model = YOLO("best.pt")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ==========================================
# 4. THE VIDEO PROCESSOR (The Brain)
# This class runs on every single video frame
# ==========================================
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.threshold = 0.5
        # We need a way to pass data back to Streamlit, but doing so 
        # from inside this callback is tricky. 
        # For now, we visualize everything on the video itself.

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # 1. Run YOLO Inference
        results = model(img, conf=0.5)
        
        # 2. Draw Boxes
        for r in results:
            img = r.plot()
            
            # 3. Simple Logic: If a sign is found, draw it big
            if len(r.boxes) > 0:
                cls_id = int(r.boxes.cls[0])
                name = r.names[cls_id].upper()
                
                # Draw the predicted text on the video frame
                cv2.putText(img, f"SIGN: {name}", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        # Return the processed frame to the browser
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ==========================================
# 5. UI LAYOUT
# ==========================================
st.title("RealSign Live Video Stream")
st.markdown("### 📹 Continuous Translation Mode")
st.info("The video below is being processed in real-time. Signs detected will be labeled on screen.")

# THE MAGIC COMPONENT
# rtc_configuration is needed for cloud deployment (STUN server)
# Define the "Phone Book" of internet connections (STUN Servers)
RTC_CONFIGURATION = {
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
        {"urls": ["stun:stun3.l.google.com:19302"]},
        {"urls": ["stun:stun4.l.google.com:19302"]},
    ]
}

# (Optional Pro Tip) If it STILL fails, you need a TURN server (Relay).
# You can get a free one from Twilio, but try the list above first.

webrtc_streamer(
    key="sign-detection",
    video_processor_factory=VideoProcessor,
    rtc_configuration=RTC_CONFIGURATION,  # <--- Use the new config here
    media_stream_constraints={"video": True, "audio": False}
)
st.divider()
st.markdown("**Note:** If the video freezes, refresh the page. Cloud GPU latency may vary.")
