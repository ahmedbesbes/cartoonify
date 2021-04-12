import time
import base64
from io import BytesIO
import requests
import streamlit as st
from PIL import Image

st.set_page_config(layout="wide")

url = "https://tbuxfdm545.execute-api.eu-west-3.amazonaws.com/dev/transform"

model_names = ["Hosoda", "Hayao", "Shinkai", "Paprika"]
model_name = st.sidebar.selectbox("Select a model", options=model_names)

load_size = st.sidebar.slider("Set image size", 100, 800, 300, 20)
uploaded_image = st.sidebar.file_uploader("Upload image", type=["png", "jpg", "jpeg"])


cols = st.beta_columns((1, 1))

with cols[0]:
    input_image = st.empty()

with cols[1]:
    transformed_image = st.empty()


if uploaded_image is not None:
    transform = st.sidebar.button("Cartoonify!")

    pil_image = Image.open(uploaded_image)
    image = base64.b64encode(uploaded_image.getvalue()).decode("utf-8")

    data = {
        "image": image,
        "model_id": model_names.index(model_name),
        "load_size": load_size,
    }

    input_image.image(uploaded_image)

    if transform:
        t0 = time.time()
        response = requests.post(url, json=data)
        delta = time.time() - t0
        image = response.json()["output"]
        image = image[image.find(",") + 1 :]
        dec = base64.b64decode(image + "===")
        binary_output = BytesIO(dec)

        st.sidebar.warning(f"Processing took {delta:.3} seconds")

        transformed_image.image(binary_output)
