import streamlit as st
import cv2
import av
import os
import uuid
import numpy as np
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# ==========================================
# 1. SETUP & PRIVACY CHECK
# ==========================================
st.set_page_config(page_title="RealSign Learning", page_icon="🎓")

if 'privacy_accepted' not in st.session_state:
    st.session_state['privacy_accepted'] = False

# Create folders for data collection if they don't exist
if not os.path.exists("training_data"):
    os.makedirs("training_data")

# --- PRIVACY MODAL ---
if not st.session_state['privacy_accepted']:
    st.markdown("## 🔒 Privacy & Learning Opt-In")
    st.info("To improve RealSign, we need your help! You can opt-in to save images where the model makes a mistake. These images are used ONLY to retrain the AI.")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ I Agree (Enable Learning)", type="primary"):
            st.session_state['privacy_accepted'] = True
            st.rerun()
    with col2:
        if st.button("❌ No, Just Use Model"):
            st.session_state['privacy_accepted'] = "OPT_OUT"
            st.rerun()
    
    st.warning("You must choose an option to continue.")
    st.stop()  # Stop the app here until they choose

# ==========================================
# 2. MODEL & CONFIG
# ==========================================
try:
    model = YOLO("best.pt")
except:
    st.error("Model not found!")
    st.stop()

# Robust Connection Config (Fixes the "Taking too long" error)
RTC_CONFIGURATION = {
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
    ]
}

# ==========================================
# 3. VIDEO PROCESSOR
# ==========================================
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.last_frame = None
        self.last_pred = "None"

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Save frame for potential training (store in memory temporarily)
        self.last_frame = img.copy()
        
        # Inference
        results = model(img, conf=0.5)
        for r in results:
            img = r.plot()
            if len(r.boxes) > 0:
                # Get the top prediction
                cls_id = int(r.boxes.cls[0])
                self.last_pred = r.names[cls_id]
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ==========================================
# 4. MAIN UI
# ==========================================
st.title("🎓 RealSign: Active Learning")

# Only show the "Improvement" tools if they opted in
if st.session_state['privacy_accepted'] == True:
    st.success("mode: 🟢 Active Learning Enabled")
else:
    st.warning("mode: 🔴 Private Mode (No Data Saved)")

ctx = webrtc_streamer(
    key="active-learning",
    video_processor_factory=VideoProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False}
)

# ==========================================
# 5. THE FEEDBACK LOOP
# ==========================================
if ctx.video_processor:
    # We grab the last known prediction from the video processor
    current_pred = ctx.video_processor.last_pred
    
    st.metric("Model Sees:", current_pred)
    
    # "Correction" Interface (Only if Opted-In)
    if st.session_state['privacy_accepted'] == True:
        with st.expander("Is the model wrong? Teach it!", expanded=True):
            st.write("If the model made a mistake, tell us the correct sign:")
            
            # Select the ACTUAL letter
            correct_label = st.selectbox("Correct Label:", ["A", "B", "C", "D", "E", "F", "Hello", "Thanks"])
            
            if st.button("📸 Snap & Save Correction"):
                if ctx.video_processor.last_frame is not None:
                    # Save the image with a unique name
                    filename = f"training_data/{correct_label}_{uuid.uuid4().hex[:8]}.jpg"
                    
                    # Convert RGB to BGR for OpenCV saving
                    save_img = cv2.cvtColor(ctx.video_processor.last_frame, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(filename, save_img)
                    
                    st.toast(f"Saved! You taught the model that this is '{correct_label}'")
                else:
                    st.error("No video frame detected yet.")
