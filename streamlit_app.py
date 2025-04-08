import streamlit as st
from PIL import Image
import numpy as np
from fractal_dimension import compute_fractal_dimension

st.title("Cancer Tissue Fractality Detector")

st.write("""
Upload a tissue image (JPG, PNG, etc.) and the app will calculate its fractal dimension.
A fractal dimension above **~1.65** is typically associated with malignant tumors.
""")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Tissue Image', use_column_width=True)
    
    
    image_resized = image.resize((256,256))
    image_np = np.array(image_resized)
    
    fd = compute_fractal_dimension(image_np)
    st.write(f"**Fractal Dimension:** {fd:.4f}")
    
    
    threshold_fd = 1.65
    if fd > threshold_fd:
        st.write("The fractal dimension is above the threshold, indicating that the tumor might be malignant.")
    else:
        st.write("The fractal dimension is below the threshold, suggesting that the tumor might be benign.")

st.write("Note: The threshold of ~1.65 is based on current experimental evidence and may require calibration for your dataset.")
