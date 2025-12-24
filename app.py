import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("cat_dog_cnn.h5")

st.title("🐶🐱 Cat vs Dog Classifier")

uploaded_file = st.file_uploader("Upload an image", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).resize((128,128))
    st.image(image, caption="Uploaded Image")

    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1,128,128,3)

    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        st.success("🐶 Dog")
    else:
        st.success("🐱 Cat")
