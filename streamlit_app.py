import streamlit as st
import torch
from PIL import Image
import numpy as np
from fractal_dimension import compute_fractal_dimension


model = torch.load("best_model.pth", map_location=torch.device("cpu"))
model.eval()  

st.title("Cancer Tissue Fractality Detector")

st.write("""
Upload a tissue image (JPG, PNG, etc.) and the app will calculate its fractal dimension.
Based on the trained CNN model using fractal dimension, the tumor will be classified as **benign** or **malignant**.
""")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Tissue Image', use_column_width=True)
    
    
    image_resized = image.resize((256, 256))
    image_np = np.array(image_resized)
    
    
    fd = compute_fractal_dimension(image_np)
    st.write(f"**Fractal Dimension:** {fd:.4f}")
    
    
    
    input_tensor = torch.tensor([[fd]], dtype=torch.float32)
    
    
    with torch.no_grad():
        output = model(input_tensor)
    
    
    
    probability = output.item()
    st.write(f"**Predicted Probability of Malignancy:** {probability:.4f}")
    
    
    if probability > 0.5:
        st.write("Based on the CNN model prediction, the tumor is classified as **malignant**.")
    else:
        st.write("Based on the CNN model prediction, the tumor is classified as **benign**.")

st.write("Note: The classification decision is based on the trained CNN model. Adjust the threshold or preprocessing if required for your dataset.")
