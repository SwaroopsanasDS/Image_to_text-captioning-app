import streamlit as st
from transformers import AutoProcessor, AutoModelForCausalLM
import torch
from PIL import Image

@st.cache_resource
def load_model():
    processor = AutoProcessor.from_pretrained("microsoft/git-base")
    model = AutoModelForCausalLM.from_pretrained("microsoft/git-base")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return processor, model, device

processor, model, device = load_model()

def generate_caption(image):
    inputs = processor(images=image, return_tensors="pt").to(device)
    generated_ids = model.generate(**inputs, max_new_tokens=50)
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_caption

# Streamlit UI
st.set_page_config(page_title="GIT-Base Image Captioning", page_icon="🖼️")
st.title("🖼️ Image Caption Generator with microsoft/git-base")
st.write("Upload an image and get a smart caption using GIT-Base model!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button("Generate Caption"):
        with st.spinner('Generating... Please wait...'):
            caption = generate_caption(image)
            st.success(f"**Generated Text:** {caption}")
