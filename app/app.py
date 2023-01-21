import streamlit as st
import numpy as np
from PIL import Image
import os, sys
print("current dirc",os.getcwd())
from main import detect_object

import shutil
import datetime as dt 

# img_name='test.jpg'


# shutil.rmtree(crnt_drctr1 + '../yolov5/runs/detect')

# img_name = "img_" + str(dt.datetime.today())

def main():
    st.title("Planogram Clustering Application")
    crnt_drctr1 = os.getcwd()
    if os.path.exists(crnt_drctr1 + '/../yolov5/runs/detect'):
        shutil.rmtree(crnt_drctr1 + '/../yolov5/runs/detect')
    
    # shutil.rmtree(crnt_drctr1 + '/../image/')
    if os.path.exists(crnt_drctr1 + '/../image/'):
        shutil.rmtree(crnt_drctr1 + '/../image/')
        os.mkdir(crnt_drctr1 + '/../image/')
    else:
        os.mkdir(crnt_drctr1 + '/../image/')
    img_name = upload_image_ui()
    button1 = st.button("make prediction")
    button2 = st.button("Re-Run")
    if button1:
        crnt_drctr1 = os.getcwd()
        # if isinstance(img_array, np.ndarray):
        try:
            img_clstr = detect_object(img_name)
        except:
            if button2:
                re_run()
        img_org = crnt_drctr1 + '/../image/' + img_name
        print("show image start:")
        st.subheader("Original Image")
        st.image(img_org) 

        st.subheader("Clustered Image")
        st.image(img_clstr) 
        print("show image end")
        if button2:
            re_run()
    if button2:
        re_run()

def upload_image_ui():
    uploaded_image = st.file_uploader("Please choose an image file", type=["png", "jpg", "jpeg"])
    crnt_drctr = os.getcwd()
    
    # print("uploaded_image name", uploaded_image.name)
    
    if uploaded_image is not None:
        try:
            image = Image.open(uploaded_image)
            img_name = uploaded_image.name
            image.save(crnt_drctr + '/../image/' + img_name)
            # st.image(uploaded_image)
            # sys.exit()
        except Exception:
            st.error("Error: Invalid image")
        # else:
        #     img_array = np.array(image)
        #     return img_array
        return img_name
def re_run():
    if st.button('Return to Main Page'):
        # os.chdir()
        print("current dirc",os.getcwd())
        st.session_state.runpage = main
        st.experimental_rerun()

if __name__ == '__main__':
    
    main()
    
