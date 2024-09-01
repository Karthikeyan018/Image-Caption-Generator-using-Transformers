import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# Add CSS for the background image using Streamlit's main content area
page_bg_img = '''
<style>
[data-testid="stAppViewContainer"] {
background-image: url("https://static.vecteezy.com/system/resources/previews/009/749/967/non_2x/wave-background-in-blue-color-with-line-elements-technology-startup-game-suitable-for-websites-mobile-applications-posters-games-printing-and-more-free-vector.jpg");
background-size: cover;
background-position: center;
background-repeat: no-repeat;
}

[data-testid="stSidebar"] {
background-image: url("https://static.vecteezy.com/system/resources/previews/009/749/967/non_2x/wave-background-in-blue-color-with-line-elements-technology-startup-game-suitable-for-websites-mobile-applications-posters-games-printing-and-more-free-vector.jpg");
background-size: cover;
background-position: center;
background-repeat: no-repeat;
} 
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)

# Load the BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

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
                if caption:
                    st.markdown(f"<p style='font-size:25px;'><strong>Caption:</strong> {caption}</p>", unsafe_allow_html=True)
                st.image(image)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please upload an image.")

