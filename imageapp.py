import streamlit as st

# Handle imports
try:
    import torch
    from PIL import Image
    from transformers import BlipProcessor, BlipForConditionalGeneration
except ModuleNotFoundError as e:
    st.error(f"Module not found: {e.name}")
    raise

# Check and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the BLIP model and processor
@st.cache_resource()
def load_model_and_processor():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    model.to(device)
    return processor, model

processor, model = load_model_and_processor()

# Function to generate caption
def generate_caption(image):
    inputs = processor(image, return_tensors="pt").to(device)
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# Streamlit app
st.title("Image Caption Generator")

# File uploader for image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Generate caption button
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
