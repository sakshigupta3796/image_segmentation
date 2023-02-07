import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import os, sys
print("start app current dirc",os.getcwd())
from main import detect_object

import shutil
import datetime as dt 


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
        
        img_clstr, clustered_images,clustered_img_name = detect_object(img_name)

        img_org = crnt_drctr1 + '/../image/' + img_name
        print("show image start:")
        new_title = '<p style="font-family:sans-serif; color:Black; font-size: 42px;">Original Image</p>'
        st.markdown(new_title, unsafe_allow_html=True)
        st.image(img_org) 

        new_title = '<p style="font-family:sans-serif; color:Black; font-size: 42px;">Clustered Image</p>'
        st.markdown(new_title, unsafe_allow_html=True)
        st.image(img_clstr) 
        print("show image end")

        new_title = '<p style="font-family:sans-serif; color:Green; font-size: 42px;">Image Cluster and their Counts</p>'
        st.markdown(new_title, unsafe_allow_html=True)

        clustered_images = dict(sorted(clustered_images.items(), reverse=True, key=lambda item: len(item[1])))

        
        lst_df = []
        image_lst= []
        count_lst= []
        pdf_df = pd.DataFrame(columns=["image", "count"])
        for key in clustered_images.keys():
            
            c1,c2 = st.columns([1,2])
            first_image = clustered_images[key][0][0]
            image_lst.append(first_image)
            print("first_image", first_image)
            c1.image(crnt_drctr1 + "/../yolov5/runs/detect/exp/crops/o/"+ first_image)

            n_rows = len(clustered_images[key])
            md_results = f"**{n_rows}**"
            count_lst.append(n_rows)
            print("count", n_rows)
            # new_title1 = '<p style="font-family:sans-serif; color:Green; font-size: 42px;">len(clustered_images[key])</p>'
            c2.markdown(md_results)
            # lst_df.append(pdf_df)
            # print("inside final image df", pdf_df)
        # final_df = pd.concat(lst_df)
        pdf_df = pd.DataFrame({"image":image_lst, "Count":count_lst})
        print("final image df", pdf_df)
        pdf_df.to_csv("clustured_image_df.csv")

        print("inside download")
        with open("clustured_image_df.csv", "rb") as file:
            st.download_button(label="Clustered Image data frame Download", data=file, file_name="clustured_image_df.csv")

        with open(crnt_drctr1 + '/../image/' + "clustered_" + clustered_img_name, "rb") as file:
                    st.download_button(label="Clustered Image Download", data=file, file_name=clustered_img_name)



        if button2:
            re_run()
        print("end:::::")
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
    # os.chdir()
    print("re-started app")
    print("current dirc",os.getcwd())
    st.session_state.runpage = main
    st.experimental_rerun()

if __name__ == '__main__':
    
    main()
    
