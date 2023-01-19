import torch # YOLOv5 implemented using pytorch
from IPython.display import Image #this is to render predictions
from PIL import Image, ImageEnhance, ImageFilter
import pandas as pd
import os
import numpy as np
from random import randint
import pickle
from PIL import Image, ImageEnhance, ImageFilter
# for loading/processing the images  
# from keras.preprocessing.image import load_img 
from tensorflow.keras.utils import load_img

from tensorflow.keras.utils import img_to_array 
from keras.applications.vgg16 import preprocess_input 

# models 
from keras.applications.vgg16 import VGG16 
from keras.models import Model


from scipy import spatial
%pylab inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.metrics import structural_similarity
import cv2

!pip install -q git+https://github.com/huggingface/transformers.git datasets
from transformers import ImageGPTFeatureExtractor, ImageGPTModel

!pip install imagehash
import imagehash


import torchvision
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes

import feature_extractor
import functions as f
import sys
sys.exit()
img_name=''
img_path=''
image=  Image.open(img_path+img_name)
!python ../yolov5/detect.py --source /content/data/SKU110K_fixed/images/test/test_1099.jpg --save-txt --save-conf --save-crop --weights /content/yolov5/runs/train/exp2/weights/best.pt

df_org=f.filter_images("/content/yolov5/runs/detect/exp/labels/"+img_name.split('.')[0]+".txt",0.8)
print(df_org.shape)
file_ls_org=df_org['file_path'].apply(lambda x:x.split('/')[-1]).to_list()

!mkdir  /content/yolov5/runs/detect/exp/crops/filter_o

path=r'/content/yolov5/runs/detect/exp/crops/o/'
path_filter=r'/content/yolov5/runs/detect/exp/crops/filter_o/'
f.filter_cropped(path,path_filter,file_ls_org)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

feature_extractor_GPT = ImageGPTFeatureExtractor.from_pretrained("openai/imagegpt-small")
model_GPT = ImageGPTModel.from_pretrained("openai/imagegpt-small")
model_GPT.to(device)

model_VGG = VGG16()
model_VGG = Model(inputs = model_VGG.inputs, outputs = model_VGG.layers[-2].output)

path = r"/content/yolov5/runs/detect/exp/crops/o/"
data_GPT_org,data_VGG_org=f.call_gpt_vgg(path,file_ls_org,feature_extractor_GPT,model_GPT,device,model_VGG)

path = r"/content/yolov5/runs/detect/exp/crops/filter_o/"
data_GPT_filter,data_VGG_filter=f.call_gpt_vgg(path,file_ls_org,feature_extractor_GPT,model_GPT,device,model_VGG)

image_path_org = "/content/yolov5/runs/detect/exp/crops/o/"
image_path_filter = "/content/yolov5/runs/detect/exp/crops/filter_o/"
t1=0.001
hashfunc=imagehash.colorhash
clusters = f.clustering_cosine_struture_similarity(data_GPT_org,data_VGG_org,data_GPT_filter,data_VGG_filter,image_path_org,image_path_filter,t1,hashfunc)

clusters.keys()

#f.plot_clusters(clusters,'/content/yolov5/runs/detect/exp/crops/o/')


# data = pd.read_csv('/content/yolov5/runs/detect/exp/labels/'+img_name.split('.)[0]+".txt", sep=" ", header=None)
# data.columns = ["classs", "x1", "y1", "x2", "y2", "path"]
#data["img_name"] = data["path"].apply(lambda x:x.split("/")[-1])
df_org["img_name"] = df_org["file_path"].apply(lambda x:x.split("/")[-1])
img=f.images_after_clustering(img_path,df_org,clusters)
print(img)
