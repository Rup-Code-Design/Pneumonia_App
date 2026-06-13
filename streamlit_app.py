import streamlit as st
import keras
import numpy as np
import cv2
from PIL import Image
from fpdf import FPDF
import io
from model_builder import build_model 

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="Pneumonia AI", page_icon="🫁")

# Initialize Session State for History
if 'history' not in st.session_state:
    st.session_state.history = []

# ==========================================
# LOAD MODEL DIRECTLY IN STREAMLIT (Cached)
# ==========================================
@st.cache_resource
def load_prediction_model():
    model = build_model(input_shape=(224, 224, 3), num_classes=1)
    model.load_weights('best_xception_model.keras')
    return model

try:
    model = load_prediction_model()
except Exception as e:
    st.error(f"Error loading model weights: {e}")

# ==========================================
# SIDEBAR & HEADER
# ==========================================
st.title("🫁 Pneumonia Detection System")
st.markdown("Upload a Chest X-ray image to get an instant diagnosis.")

with st.sidebar:
    st.header("About the Project")
    st.info("Uses Xception + Residual Blocks model for X-ray analysis.")
    st.write("Built completely within Streamlit Cloud.")
    st.divider()
    st.header("Recent Scans")
    if not st.session_state.history:
        st.write("No scans yet.")
    for item in st.session_state.history:
        st.text(item)

# ==========================================
# FILE UPLOADER & CORE LOGIC
# ==========================================
uploaded_file = st.file_uploader("Choose an x-ray image...", type=["jpg", "png", "jpeg"])
res_class = None

if uploaded_file is not None:
    try:
        file_bytes = uploaded_file.getvalue()
        bytes_stream = io.BytesIO(file_bytes)
        image = Image.open(bytes_stream)
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(image, caption='Uploaded Image', use_container_width=True)
        
        with col2:
            if st.button("Analyze Image"):
                with st.spinner('Analyzing locally...'):
                    
                    nparr = np.frombuffer(file_bytes, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if img is None:
                        st.error("Could not decode the uploaded image.")
                    else:
                        is_valid_chest_xray = True
                        
                        # 1. Color Saturation Check
                        b, g, r = cv2.split(img)
                        color_diff = np.mean(np.abs(b.astype(np.float32) - g.astype(np.float32))) + \
                                     np.mean(np.abs(g.astype(np.float32) - r.astype(np.float32)))
                        if color_diff > 9.0:
                            is_valid_chest_xray = False
                        
                        # 2. Structural & Spatial Variance Check
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        h, w = gray.shape
                        
                        img_variance = np.var(gray)
                        _, thresh = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
                        active_pixel_ratio = np.sum(thresh == 255) / (h * w)
                        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
                        
                        if img_variance < 800 or img_variance > 5800:
                            is_valid_chest_xray = False
                            
                        if active_pixel_ratio < 0.25 or active_pixel_ratio > 0.92:
                            is_valid_chest_xray = False
                            
                        if blur_score < 10.0 or blur_score > 1200.0:
                            is_valid_chest_xray = False

                        if not is_valid_chest_xray:
                            res_class = "Rejected: Invalid/Non-Chest X-ray Image"
                            st.error(f"❌ {res_class}")
                            st.warning("Please upload a valid CHEST X-ray image.")
                        else:
                            img_resized = cv2.resize(img, (224, 224))
                            img_normalized = img_resized.astype(np.float32) / 255.0
                            img_input = np.expand_dims(img_normalized, axis=0)
                            
                            prediction = model.predict(img_input)
                            confidence_scores = prediction[0]
                            
                            normal_score = float(confidence_scores[0])
                            pneumonia_score = float(confidence_scores[1])
                            
                            if pneumonia_score > normal_score:
                                res_class = "Pneumonia"
                                st.error(f"Diagnosis: {res_class}")
                                st.warning("Note: Pneumonia indicators detected. Please consult a radiologist.")
                            else:
                                res_class = "Normal"
                                st.success(f"Diagnosis: {res_class}")
                                st.info("Note: No clinical signs of Pneumonia detected.")
                            
                            entry = f"{res_class} - {uploaded_file.name}"
                            if entry not in st.session_state.history:
                                st.session_state.history.append(entry)
                                st.rerun()
                        
                        if res_class and "Rejected" not in res_class:
                            clean_res_class = str(res_class).replace("🫁", "").replace("❌", "").strip()
                            clean_filename = str(uploaded_file.name).encode('ascii', 'ignore').decode('ascii')
                            
                            pdf = FPDF()
                            pdf.add_page()
                            
                            pdf.set_font("Arial", 'B', 18)
                            pdf.cell(200, 15, txt="Pneumonia AI Diagnostic Report", ln=True, align='C')
                            pdf.set_line_width(0.5)
                            pdf.line(10, 25, 200, 25)
                            pdf.ln(10)
                            
                            pdf.set_font("Arial", 'B', 12)
                            pdf.cell(40, 10, txt="File Name:", ln=False)
                            pdf.set_font("Arial", size=12)
                            pdf.cell(150, 10, txt=clean_filename, ln=True)
                            
                            pdf.set_font("Arial", 'B', 12)
                            pdf.cell(40, 10, txt="Analysis Result:", ln=False)
                            pdf.set_font("Arial", size=12)
                            pdf.cell(150, 10, txt=clean_res_class, ln=True)
                            
                            pdf.ln(15)
                            pdf.set_font("Arial", 'I', 10)
                            pdf.cell(200, 10, txt="Disclaimer: AI-generated report. Please consult a medical practitioner.", ln=True, align='L')
                            
                            pdf_output = pdf.output(dest='S').encode('latin-1', errors='ignore')
                            
                            st.divider()
                            st.download_button(
                                label="Download Report", 
                                data=pdf_output, 
                                file_name=f"Report_{clean_filename}.pdf", 
                                mime="application/pdf"
                            )
                            
    except Exception as img_err:
        st.error(f"Error processing app logic: {img_err}")