import torch # YOLOv5 implemented using pytorch
from IPython.display import Image #this is to render predictions
from PIL import Image, ImageEnhance, ImageFilter
import pandas as pd
import os
import numpy as np
from random import randint
import pickle

# models 
from keras.applications.vgg16 import VGG16 
from keras.models import Model

from transformers import ImageGPTFeatureExtractor, ImageGPTModel
from torchvision.io import read_image
import imagehash
import shutil

import functions as f
import sys

sys.path.append("../yolov5")
import detect as d


def detect_object(img_name):

    image_path="../image/" + img_name
    weights="best.pt"

    def object_detection( image_path, weights):
        d.run(source= image_path, weights=weights, save_txt=True, save_crop=True,save_conf=True )    
    crnt_drctr = os.getcwd()
    # shutil.rmtree(crnt_drctr + '../yolov5/runs/detect')
    object_detection(image_path,weights)
    os.mkdir(crnt_drctr + '/../yolov5/runs/detect/exp/crops/filter_o/')

    df_org=f.filter_images("../yolov5/runs/detect/exp/labels/"+img_name.split('.')[0]+".txt",0.8)
    print(df_org.shape)
    file_ls_org=df_org['file_path'].apply(lambda x:x.split('/')[-1]).to_list()

    

    path= crnt_drctr + '/../yolov5/runs/detect/exp/crops/o/'
    path_filter= crnt_drctr + '/../yolov5/runs/detect/exp/crops/filter_o/'
    f.filter_cropped(path,path_filter,file_ls_org)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feature_extractor_GPT = ImageGPTFeatureExtractor.from_pretrained("openai/imagegpt-small")
    model_GPT = ImageGPTModel.from_pretrained("openai/imagegpt-small")
    model_GPT.to(device)

    model_VGG = VGG16()
    model_VGG = Model(inputs = model_VGG.inputs, outputs = model_VGG.layers[-2].output)


    data_GPT_org,data_VGG_org=f.call_gpt_vgg(path,file_ls_org,feature_extractor_GPT,model_GPT,device,model_VGG)


    data_GPT_filter,data_VGG_filter=f.call_gpt_vgg(path_filter,file_ls_org,feature_extractor_GPT,model_GPT,device,model_VGG)

    image_path_org = path 
    image_path_filter = path_filter
    t1=0.001
    hashfunc=imagehash.colorhash
    clusters = f.clustering_cosine_struture_similarity(data_GPT_org,data_VGG_org,data_GPT_filter,data_VGG_filter,image_path_org,image_path_filter,t1,hashfunc)

    # clusters.keys()


    image_org= read_image(crnt_drctr + '/../image/' + img_name)
    df_org["img_name"] = df_org["file_path"].apply(lambda x:x.split("/")[-1])
    img=f.images_after_clustering(image_path, image_org, df_org, clusters)
    img.save(crnt_drctr + '/../image/' + "clustered_" + img_name)

    return img

