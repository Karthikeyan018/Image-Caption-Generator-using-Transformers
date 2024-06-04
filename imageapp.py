import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

# Cache model and processor to reduce memory usage
@st.cache_resource()
def load_model_and_processor():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, model = load_model_and_processor()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Generate caption function
def generate_caption(image):
    inputs = processor(image, return_tensors="pt").to(device)
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# Streamlit app
st.title("Image Caption Generator")

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if st.button("Generate Caption"):
    if uploaded_image is not None:
        with st.spinner("Generating caption..."):
            try:
                image = Image.open(uploaded_image).convert("RGB")
                caption = generate_caption(image)
                st.success("Caption generated!")
                st.image(image, caption=caption)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please upload an image.")
