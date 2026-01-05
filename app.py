import streamlit as st
import cv2
from ultralytics import YOLO
from PIL import Image, ImageOps
import numpy as np

# ==========================================
# 1. PAGE CONFIGURATION
# Standard enterprise configuration with no icons.
# ==========================================
st.set_page_config(
    page_title="RealSign",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. PROFESSIONAL CSS STYLING
# A clean, corporate dark theme (Slate Blue/Grey).
# ==========================================
st.markdown("""
    <style>
    /* Global Font Settings */
    html, body, [class*="css"] {
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }

    /* Main Background */
    .stApp {
        background-color: #0e1117;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #262730;
        border-right: 1px solid #464b5c;
    }

    /* Headers */
    h1, h2, h3 {
        color: #ffffff;
        font-weight: 600;
    }

    /* Result Card Styling */
    .metric-card {
        background-color: #1f2937;
        border: 1px solid #374151;
        border-radius: 8px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
    }

    /* Prediction Text */
    .prediction-header {
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #9ca3af;
        margin-bottom: 8px;
    }
    
    .prediction-value {
        font-size: 36px;
        font-weight: 700;
        color: #60a5fa; /* Professional Blue */
        margin: 0;
    }

    .confidence-text {
        font-size: 14px;
        color: #9ca3af;
        margin-top: 8px;
    }

    /* Hide default Streamlit menu for cleaner look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 3. MODEL LOADING
# ==========================================
@st.cache_resource
def load_model():
    return YOLO("best.pt")

try:
    model = load_model()
except Exception as e:
    st.error(f"System Error: Model failed to load. Details: {e}")
    st.stop()

# ==========================================
# 4. SIDEBAR NAVIGATION
# ==========================================
st.sidebar.title("RealSign")
st.sidebar.markdown("---")
st.sidebar.subheader("Configuration")

# Clean, text-only navigation
app_mode = st.sidebar.radio("Input Source", ["Live Camera", "Image Upload"])

st.sidebar.markdown("---")
st.sidebar.markdown("**System Status:** Online")
st.sidebar.markdown("**Model Version:** YOLOv11n")

# ==========================================
# 5. MAIN INTERFACE LOGIC
# ==========================================
st.title("Real-Time Sign Language Recognition")
st.markdown("Automated gesture recognition system powered by computer vision.")
st.divider()

def display_inference_results(img, result):
    """
    Renders the analysis results in a structured format.
    """
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.image(img, caption="Processed Input", use_container_width=True)
        
    with col2:
        if len(result[0].boxes) > 0:
            # Extract data
            cls_id = int(result[0].boxes.cls[0])
            name = result[0].names[cls_id]
            conf = float(result[0].boxes.conf[0])
            
            # Display Professional Metric Card
            st.markdown(f"""
            <div class="metric-card">
                <div class="prediction-header">Detected Class</div>
                <div class="prediction-value">{name.upper()}</div>
                <div class="confidence-text">Confidence Score: {conf:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Empty State
            st.markdown("""
            <div class="metric-card" style="border-color: #7f1d1d;">
                <div class="prediction-header" style="color: #f87171;">Status</div>
                <div class="prediction-value" style="color: #f87171; font-size: 24px;">No Detection</div>
                <div class="confidence-text">Ensure hand is visible within frame.</div>
            </div>
            """, unsafe_allow_html=True)

# --- MODE 1: LIVE CAMERA ---
if app_mode == "Live Camera":
    st.subheader("Camera Input")
    img_file = st.camera_input("Capture Frame")
    
    if img_file:
        img = Image.open(img_file)
        img = ImageOps.mirror(img)
        
        # Inference
        results = model(img)
        res_plotted = results[0].plot()
        
        display_inference_results(res_plotted, results)

# --- MODE 2: UPLOAD IMAGE ---
elif app_mode == "Image Upload":
    st.subheader("File Upload")
    uploaded_file = st.file_uploader("Select an image file", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        img = Image.open(uploaded_file)
        results = model(img)
        res_plotted = results[0].plot()
        display_inference_results(res_plotted, results)