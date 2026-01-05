import streamlit as st
import cv2
from ultralytics import YOLO
from PIL import Image, ImageOps
import numpy as np
import time

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="RealSign Pro",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. SESSION STATE (The "Brain" / Memory)
# This keeps track of the sentence even when the app reloads
# ==========================================
if 'sentence' not in st.session_state:
    st.session_state['sentence'] = ""

if 'last_prediction' not in st.session_state:
    st.session_state['last_prediction'] = None

# ==========================================
# 3. CUSTOM CSS (UI/UX Upgrade)
# ==========================================
st.markdown("""
    <style>
    .stApp {background-color: #0e1117;}
    
    /* Main Output Display (The "Chat Bubble") */
    .sentence-box {
        background-color: #1f2937;
        border: 2px solid #374151;
        border-radius: 12px;
        padding: 20px;
        font-size: 28px;
        color: #ffffff;
        font-family: 'Courier New', monospace;
        margin-bottom: 20px;
        min-height: 80px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    /* Confidence Meter styling */
    .stProgress > div > div > div > div {
        background-color: #60a5fa;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 4. LOAD MODEL
# ==========================================
@st.cache_resource
def load_model():
    return YOLO("best.pt")

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ==========================================
# 5. SIDEBAR CONTROLS
# ==========================================
st.sidebar.title("RealSign 🤟")
st.sidebar.markdown("---")
app_mode = st.sidebar.radio("Mode", ["Live Translator", "Single Detection"])
st.sidebar.markdown("---")

# Sensitivity Control
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.4, 0.05)
st.sidebar.info("Adjust threshold if detections are flickering.")

# ==========================================
# 6. MAIN FUNCTIONS
# ==========================================
def update_sentence(letter):
    """Adds a letter to the sentence."""
    st.session_state['sentence'] += letter

def backspace():
    """Removes last character."""
    st.session_state['sentence'] = st.session_state['sentence'][:-1]

def clear_sentence():
    """Clears everything."""
    st.session_state['sentence'] = ""

def add_space():
    """Adds a space."""
    st.session_state['sentence'] += " "

# ==========================================
# 7. UI LAYOUT
# ==========================================
st.title("RealSign Translator")

# --- TOP SECTION: THE LIVE TRANSLATION DISPLAY ---
st.markdown("### 📝 Translation Output")
st.markdown(f'<div class="sentence-box">{st.session_state["sentence"] if st.session_state["sentence"] else "Waiting for input..."}</div>', unsafe_allow_html=True)

# Control Buttons Row
col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("Space ␣", use_container_width=True):
        add_space()
with col2:
    if st.button("Backspace ⌫", use_container_width=True):
        backspace()
with col3:
    if st.button("Clear 🗑️", use_container_width=True):
        clear_sentence()
with col4:
    # Basic "Speak" alert (Full TTS requires extra libraries like gTTS)
    if st.button("🔊 Read Aloud", use_container_width=True):
        if st.session_state['sentence']:
            st.toast(f"Speaking: {st.session_state['sentence']}")
            # Note: Browser TTS requires Javascript, which is complex in Streamlit.
            # This is a visual placeholder for the UX element.

st.divider()

# --- BOTTOM SECTION: CAMERA INPUT & PROCESSING ---
if app_mode == "Live Translator":
    col_cam, col_detail = st.columns([2, 1])
    
    with col_cam:
        # Camera Input
        img_file = st.camera_input("Capture Sign", label_visibility="collapsed")
    
    if img_file:
        img = Image.open(img_file)
        img = ImageOps.mirror(img)
        
        # Inference
        results = model(img, conf=conf_threshold)
        res_plotted = results[0].plot()
        
        # Display Result
        with col_cam:
            st.image(res_plotted, caption="Analyzed Frame", use_container_width=True)
            
        # Logic to add letter
        if len(results[0].boxes) > 0:
            cls_id = int(results[0].boxes.cls[0])
            name = results[0].names[cls_id].upper()
            conf = float(results[0].boxes.conf[0])
            
            with col_detail:
                st.markdown("#### Detected:")
                st.markdown(f"<h1 style='color:#60a5fa;'>{name}</h1>", unsafe_allow_html=True)
                st.progress(conf, text=f"Confidence: {int(conf*100)}%")
                
                # "Add to Sentence" Button (User confirms the sign)
                # We use a button here to prevent random flickering letters from ruining the sentence
                if st.button(f"Add '{name}' to Text", type="primary", use_container_width=True):
                    update_sentence(name)
                    st.rerun() # Refresh to update the top box immediately

elif app_mode == "Single Detection":
    st.write("Simple detection mode (No sentence building).")
    img_file = st.camera_input("Take Picture")
    if img_file:
        img = Image.open(img_file)
        results = model(img)
        st.image(results[0].plot(), use_container_width=True)