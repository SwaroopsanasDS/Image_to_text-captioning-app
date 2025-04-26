# --------- IMPORTS ----------
import streamlit as st
from transformers import AutoProcessor, AutoModelForCausalLM
import torch
from PIL import Image
import time

# --------- CONFIGURE PAGE ----------
st.set_page_config(
    page_title="Futuristic Image Caption Generator",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --------- CUSTOM STYLE WITH ANIMATIONS ----------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
        color: #222222;
    }
    .stApp {
        background-color: white;
        color: #333;
    }
    h1, h2, h3 {
        color: #0A84FF; /* Professional attractive blue */
        font-weight: 600;
        animation: fadeIn 1.5s ease-in-out;
    }
    h2 {
        animation: fadeIn 2s ease-in-out;
    }
    .stButton>button {
        background-color: #0A84FF;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.7em 1.2em;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease-in-out;
        box-shadow: 0 0 10px rgba(0,229,255,0.5);
    }
    .stButton>button:hover {
        background-color: #0066CC;
        color: white;
        box-shadow: 0 0 15px rgba(0,184,212,0.9);
    }
    .css-1cpxqw2 { 
        background-color: #fff;
        border: 1px solid #0A84FF;
    }
    /* Fade-in Animation */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    /* Bounce-in Animation for Image */
    @keyframes bounceIn {
        0% { transform: scale(0); }
        60% { transform: scale(1.1); }
        100% { transform: scale(1); }
    }
    .spinner {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .spinner div {
        margin: 3px;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background-color: #0A84FF;
        animation: bounce 0.8s infinite alternate;
    }
    .spinner div:nth-child(1) { animation-delay: 0s; }
    .spinner div:nth-child(2) { animation-delay: 0.1s; }
    .spinner div:nth-child(3) { animation-delay: 0.2s; }
    @keyframes bounce {
        0% { transform: translateY(0); }
        100% { transform: translateY(-15px); }
    }
    </style>
""", unsafe_allow_html=True)

# --------- LOAD MODEL ----------
@st.cache_resource
def load_model():
    processor = AutoProcessor.from_pretrained("microsoft/git-base")
    model = AutoModelForCausalLM.from_pretrained("microsoft/git-base")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return processor, model, device

processor, model, device = load_model()

# --------- CAPTION FUNCTION ----------
def generate_caption(image):
    inputs = processor(images=image, return_tensors="pt").to(device)
    generated_ids = model.generate(**inputs, max_new_tokens=50)
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_caption

# --------- MAIN UI ----------
st.markdown("<h1 align='center'>Image Captioning</h1>", unsafe_allow_html=True)
st.markdown("<h3 align='center' class='fade-in-text'>Upload your image and let AI describe it intelligently</h3>", unsafe_allow_html=True)
st.markdown("---")

# IMAGE UPLOAD WITH BOUNCE ANIMATION
uploaded_file = st.file_uploader("📁 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='🖼️ Uploaded Image', width=500, output_format="JPEG", channels="RGB", clamp=True)
    st.markdown("<p style='text-align: center;'>💫 Image Uploaded!</p>", unsafe_allow_html=True)

    # BUTTON FOR CAPTION GENERATION
    if st.button("✨ Generate Caption"):
        with st.spinner('Generating Caption... please wait 🚀'):
            time.sleep(1.5)
            caption = generate_caption(image)
        
        st.success("✅ Caption Generated!")
        st.markdown(f"<h2 style='text-align: center; color: #0A84FF;'>{caption}</h2>", unsafe_allow_html=True)

st.markdown("---")
st.markdown("<p style='text-align: center; font-size: 14px;'>© 2025 | Designed by Swaroop 🚀</p>", unsafe_allow_html=True)
