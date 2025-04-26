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

# --------- CUSTOM CLASSY + GLOW CSS ----------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
        background-color: #0d0d0d;
        color: #e0e0e0;
    }
    .stApp {
        background: linear-gradient(135deg, #0a0a0a, #1a1a1a);
    }
    h1, h2, h3, h4, p, li {
        color: #E0F7FA;
        text-shadow: 0px 0px 8px rgba(0, 255, 255, 0.6);
        font-weight: 600;
    }
    .stButton>button {
        background-color: #00e5ff;
        color: black;
        border: none;
        border-radius: 8px;
        padding: 0.75em 1.5em;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease-in-out;
        box-shadow: 0 0 10px rgba(0,229,255,0.7);
    }
    .stButton>button:hover {
        background-color: #00b8d4;
        color: white;
        box-shadow: 0 0 20px rgba(0,184,212,0.9);
    }
    .css-1cpxqw2 { 
        background-color: #1c1c1c;
        border: 1px solid #00e5ff;
    }
    .uploadedImage {
        border-radius: 10px;
        box-shadow: 0px 0px 15px rgba(0, 255, 255, 0.2);
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
st.markdown("<h1 align='center'>🚀 AI Image Captioning</h1>", unsafe_allow_html=True)
st.markdown("<h3 align='center'>Upload an image and let AI describe it smartly.</h3>", unsafe_allow_html=True)
st.markdown("---")

uploaded_file = st.file_uploader("📁 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True, output_format="JPEG", channels="RGB", clamp=True)

    if st.button("⚡ Generate Caption"):
        with st.spinner('Thinking... 🚀'):
            time.sleep(1.5)
            caption = generate_caption(image)
        
        st.success("✅ Caption Generated!")
        st.markdown(f"<h2 style='text-align: center; color: #00e5ff; text-shadow: 0px 0px 10px rgba(0, 229, 255, 0.8);'>{caption}</h2>", unsafe_allow_html=True)

st.markdown("---")
st.markdown("<p style='text-align: center; font-size: 14px;'>© 2025 | Crafted by Swaroop ✨</p>", unsafe_allow_html=True)
