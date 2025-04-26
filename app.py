# ---------- IMPORTS ----------
import streamlit as st
from transformers import AutoProcessor, AutoModelForCausalLM
import torch
from PIL import Image
import time

# ---------- CONFIGURATION ----------
st.set_page_config(
    page_title="🚀 Futuristic Image Captioner | GIT-Base",
    page_icon="🖼️",
    layout="wide",  # full width
    initial_sidebar_state="collapsed"  # hide sidebar for clean look
)

# ---------- CUSTOM CSS FOR FUTURISTIC VIBES ----------
st.markdown("""
    <style>
    body {
        background-color: #0f0f1a;
        color: #f8f8f2;
    }
    .stApp {
        background-image: linear-gradient(135deg, #0f0f1a, #1a1a2e, #16213e, #0f3460);
        background-size: 400% 400%;
        animation: gradientMove 15s ease infinite;
    }
    @keyframes gradientMove {
        0%{background-position:0% 50%}
        50%{background-position:100% 50%}
        100%{background-position:0% 50%}
    }
    h1, h2, h3 {
        color: #00f9ff;
        text-shadow: 0 0 10px #00f9ff;
        font-family: 'Orbitron', sans-serif;
    }
    .stButton>button {
        background-color: #7209b7;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 12em;
        font-size: 18px;
        font-weight: bold;
        box-shadow: 0 0 10px #7209b7;
    }
    .stButton>button:hover {
        background-color: #3a0ca3;
        box-shadow: 0 0 20px #3a0ca3;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model():
    processor = AutoProcessor.from_pretrained("microsoft/git-base")
    model = AutoModelForCausalLM.from_pretrained("microsoft/git-base")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return processor, model, device

processor, model, device = load_model()

# ---------- CAPTION FUNCTION ----------
def generate_caption(image):
    inputs = processor(images=image, return_tensors="pt").to(device)
    generated_ids = model.generate(**inputs, max_new_tokens=50)
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_caption

# ---------- APP UI ----------
st.markdown("<h1 align='center'>🚀 Futuristic Image Captioning App</h1>", unsafe_allow_html=True)
st.markdown("<h3 align='center'>Upload your image & watch AI describe it like magic! ✨</h3>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='🎯 Uploaded Image', use_column_width=True)

    if st.button("⚡ Generate Caption"):
        with st.spinner('Processing through hyperspace... 🚀✨'):
            time.sleep(2)  # Just for better UX
            caption = generate_caption(image)
        
        st.success("✅ Caption Generated!", icon="🔥")
        st.markdown(f"<h2 style='text-align: center; color: #ff007f;'>{caption}</h2>", unsafe_allow_html=True)

# ---------- FOOTER ----------
st.markdown("<hr style='border:1px solid #00f9ff;'>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Made with 💙 by Swaroop ⚡</p>", unsafe_allow_html=True)
