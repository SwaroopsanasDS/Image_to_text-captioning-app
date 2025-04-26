# -------- IMPORTS ----------
import streamlit as st
from transformers import AutoProcessor, AutoModelForCausalLM
import torch
from PIL import Image
import time

# -------- CONFIGURE PAGE ----------
st.set_page_config(
    page_title="Futuristic Image Caption Generator",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------- CUSTOM FUTURISTIC + CLASSY STYLE ----------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;600&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Poppins', sans-serif;
        background-color: #121212;
        color: #E0E0E0;
    }
    .stApp {
        background: linear-gradient(135deg, #0a0a0a, #1c1c1c);
    }
    h1, h2, h3, h4 {
        color: #00FFFF;
        font-weight: 600;
    }
    .stButton>button {
        background-color: #00FFFF;
        color: #000;
        border: none;
        border-radius: 8px;
        padding: 0.75em 1.5em;
        font-weight: 600;
        font-size: 16px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #00cccc;
        color: white;
    }
    .css-1cpxqw2 { /* upload button color */
        background-color: #1c1c1c;
        border: 1px solid #00FFFF;
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

# --------- UI SECTION ----------
st.markdown("<h1 align='center'>🚀 AI Powered Image Captioning</h1>", unsafe_allow_html=True)
st.markdown("<h3 align='center'>Upload an image, and our futuristic model describes it intelligently.</h3>", unsafe_allow_html=True)
st.markdown("---")

uploaded_file = st.file_uploader("📁 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='🖼️ Uploaded Image', use_column_width=True)

    if st.button("✨ Generate Caption"):
        with st.spinner('Analyzing image... Please wait 🚀'):
            time.sleep(1.5)
            caption = generate_caption(image)
        
        st.success("✅ Caption Generated!")
        st.markdown(f"<h2 style='text-align: center; color: #00FFFF;'>{caption}</h2>", unsafe_allow_html=True)

st.markdown("---")
st.markdown("<p style='text-align: center; font-size: 14px;'>© 2025 | Built with 🚀 by Swaroop</p>", unsafe_allow_html=True)
