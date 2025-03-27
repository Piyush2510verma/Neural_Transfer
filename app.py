import streamlit as st
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io

# Load pre-trained model
@st.cache_resource
def load_model():
    return hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

model = load_model()

# Function to preprocess the image
def load_image(img):
    img = tf.image.convert_image_dtype(img, tf.float32)  # Convert to [0, 1] range
    img = tf.image.resize(img, (256, 256))               # Resize to 256x256
    img = img[tf.newaxis, :]                             # Add batch dimension
    return img

# Streamlit UI
st.title('🎨 Neural Style Transfer')
st.write('Upload a **content image** and a **style image** to create a new stylized image.')

# Upload content and style images
content_file = st.file_uploader("Upload Content Image", type=["jpg", "jpeg", "png"])
style_file = st.file_uploader("Upload Style Image", type=["jpg", "jpeg", "png"])

if content_file and style_file:
    # Load and display input images
    content_image = Image.open(content_file).convert("RGB")
    style_image = Image.open(style_file).convert("RGB")

    st.subheader('📸 Content Image')
    st.image(content_image, use_container_width=True)

    st.subheader('🖼️ Style Image')
    st.image(style_image, use_container_width=True)

    # Convert PIL images to tensors
    content_tensor = load_image(tf.convert_to_tensor(np.array(content_image)))
    style_tensor = load_image(tf.convert_to_tensor(np.array(style_image)))

    # Style Transfer
    st.write('✨ **Generating stylized image...**')
    stylized_image = model(tf.constant(content_tensor), tf.constant(style_tensor))[0]

    # Convert tensor to image format
    result_image = np.squeeze(stylized_image) * 255
    result_image = np.array(result_image, dtype=np.uint8)

    # Display Result
    st.subheader('🌅 Stylized Image')
    st.image(result_image, use_container_width=True)

    # Download option
    result_pil = Image.fromarray(result_image)
    buf = io.BytesIO()
    result_pil.save(buf, format="JPEG")
    byte_im = buf.getvalue()

    st.download_button(
        label="💾 Download Stylized Image",
        data=byte_im,
        file_name="stylized_image.jpg",
        mime="image/jpeg"
    )

# Footer
st.write('---')
st.write('🤖 Built with TensorFlow, TensorFlow Hub, and Streamlit')

