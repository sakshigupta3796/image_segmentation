import streamlit as st
import numpy as np
from PIL import Image
from main import detect_object
import os, sys

img_name='test.jpg'


def main():
    st.title("Object detection with YOLOv4")
    upload_image_ui()

    # if isinstance(img_array, np.ndarray):

    img_clstr = detect_object(img_name)
    st.image(img_clstr) 

def upload_image_ui():
    uploaded_image = st.file_uploader("Please choose an image file", type=["png", "jpg", "jpeg"])
    crnt_drctr = os.getcwd()
    
    
    if uploaded_image is not None:
        try:
            image = Image.open(uploaded_image)
            image.save(crnt_drctr + '/../image/' + img_name)
            # sys.exit()
        except Exception:
            st.error("Error: Invalid image")
        # else:
        #     img_array = np.array(image)
        #     return img_array

if __name__ == '__main__':
    main()