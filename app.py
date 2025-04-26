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

# --------- CUSTOM STYLE (default white background, professional font) ----------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
        color: #222222;
    }
    h1, h2, h3 {
        color: #0A84FF; /* Attractive blue */
        font-weight: 700;
    }
    p, li {
        font-size: 16px;
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
    }
    .stButton>button:hover {
        background-color: #0066CC;
        color: white;
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
st.markdown("<h1 align='center'>🚀 Futuristic Image Captioning</h1>", unsafe_allow_html=True)
st.markdown("<h3 align='center'>Upload your image & let AI describe it intelligently</h3>", unsafe_allow_html=True)
st.markdown("---")

uploaded_file = st.file_uploader("📁 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='🖼️ Uploaded Image', width=500)  # ✅ Decent size now

    if st.button("✨ Generate Caption"):
        with st.spinner('Generating smart caption... 🚀'):
            time.sleep(1.5)
            caption = generate_caption(image)

        st.success("✅ Caption Generated!")
        st.markdown(f"<h2 style='text-align: center; color: #0A84FF;'>{caption}</h2>", unsafe_allow_html=True)

st.markdown("---")
st.markdown("<p style='text-align: center; font-size: 14px;'>© 2025 | Designed by Swaroop 🚀</p>", unsafe_allow_html=True)
