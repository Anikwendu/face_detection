import streamlit as st
import cv2
from PIL import Image, ImageEnhance
import numpy as np
import os

face_cascade = cv2.CascadeClassifier("C:\\Users\\Amarachi Uzochukwu\\Documents\\haarcascade_frontalface_default.xml")
path = 'C:\\Users\\Amarachi Uzochukwu\\Desktop'
def detect_face(image, min_neighbor, scale_factor, rectangle_color):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbor)
    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), rectangle_color, 2)
    return image
    cv2.imwrite(os.path.join(path, 'image.png'), faces)

def app():
    st.title("Face Detection using Viola-Jones Algorithm")
    st.write("upload image with 'file uploader'")
    st.write("adjust parameters with the sliders below")
    st.write("use color picker to choose rectangle color")
    st.write("use the detect button to detect faces")
    st.write("save images with the 'save image' button")

    min_neighbor = st.sidebar.slider("minNeighbors", min_value=1, max_value=10, value=2)
    scale_factor = st.sidebar.slider("scaleFactor", min_value=1.1, max_value=1.5, step=0.1, value=1.2)
    rectangle_color = st.sidebar.color_picker("Rectangle Color", "#0000FF")
    image_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])

    if image_file is not None:
            our_image = 'input_image.jpg'
            img = cv2.imdecode(np.fromstring(image_file.read(),np.uint8), 1)
            st.text("Original Image")
            # st.write(type(our_image))
            st.image(img)


    # Add a button to start detecting faces
    if st.button("Detect Faces"):
        image_faces = detect_face(img, min_neighbor, scale_factor, rectangle_color )
        st.image(image_faces, channels="BGR", caption="Image With Detected Faces", use_column_width=True)


if __name__ == "__main__":
    app()