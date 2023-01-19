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

import functions as f
import sys

sys.path.append("../yolov5")
# import ../yolov5/
import detect as d

# sys.exit()
img_name='test_1001.jpg'
# img_path=''
# image=  Image.open(img_path+img_name)
image_path="../image/test_1001.jpg"
weights="best.pt"

def object_detection( image_path, weights):
    d.run(source= image_path, weights=weights, save_txt=True, save_crop=True,save_conf=True )    

# object_detection(image_path,weights)

# sys.exit()
# os.system("python /mnt/d/image_clustering/image_segmentation/yolov5/detect.py --source /content/data/SKU110K_fixed/images/test/test_1099.jpg --save-txt --save-conf --save-crop --weights /mnt/d/image_clustering/image_segmentation/yolov5/runs/train/exp2/weights/best.pt")

# !python ../yolov5/detect.py --source /content/data/SKU110K_fixed/images/test/test_1099.jpg --save-txt --save-conf --save-crop --weights /content/yolov5/runs/train/exp2/weights/best.pt

df_org=f.filter_images("../yolov5/runs/detect/exp10/labels/"+img_name.split('.')[0]+".txt",0.8)
print(df_org.shape)
file_ls_org=df_org['file_path'].apply(lambda x:x.split('/')[-1]).to_list()

# os.mkdir("/mnt/d/image_clustering/image_segmentation/yolov5/runs/detect/exp10/crops/filter_o/")
print(os.getcwd())
# sys.path.append("../yolov5/")
os.chdir("../yolov5/")
print("path after:------",os.getcwd())
path='/mnt/d/image_clustering/image_segmentation/yolov5/runs/detect/exp10/crops/o/'
path_filter='/mnt/d/image_clustering/image_segmentation/yolov5/runs/detect/exp10/crops/filter_o/'
f.filter_cropped(path,path_filter,file_ls_org)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

feature_extractor_GPT = ImageGPTFeatureExtractor.from_pretrained("openai/imagegpt-small")
model_GPT = ImageGPTModel.from_pretrained("openai/imagegpt-small")
model_GPT.to(device)

model_VGG = VGG16()
model_VGG = Model(inputs = model_VGG.inputs, outputs = model_VGG.layers[-2].output)

# path = r"../yolov5/runs/detect/exp10/crops/o/"
data_GPT_org,data_VGG_org=f.call_gpt_vgg(path,file_ls_org,feature_extractor_GPT,model_GPT,device,model_VGG)

# path = r"../yolov5/runs/detect/exp10/crops/filter_o/"
data_GPT_filter,data_VGG_filter=f.call_gpt_vgg(path_filter,file_ls_org,feature_extractor_GPT,model_GPT,device,model_VGG)

image_path_org = path #"../yolov5/runs/detect/exp10/crops/o/"
image_path_filter = path_filter#"../yolov5/runs/detect/exp10/crops/filter_o/"
t1=0.001
hashfunc=imagehash.colorhash
clusters = f.clustering_cosine_struture_similarity(data_GPT_org,data_VGG_org,data_GPT_filter,data_VGG_filter,image_path_org,image_path_filter,t1,hashfunc)

clusters.keys()

#f.plot_clusters(clusters,'/content/yolov5/runs/detect/exp/crops/o/')


# data = pd.read_csv('/content/yolov5/runs/detect/exp/labels/'+img_name.split('.)[0]+".txt", sep=" ", header=None)
# data.columns = ["classs", "x1", "y1", "x2", "y2", "path"]
#data["img_name"] = data["path"].apply(lambda x:x.split("/")[-1])
image_org= read_image("/mnt/d/image_clustering/image_segmentation/image/test_1001.jpg")
df_org["img_name"] = df_org["file_path"].apply(lambda x:x.split("/")[-1])
print(df_org.head())
img=f.images_after_clustering(image_path, image_org, df_org, clusters)
img.save("/mnt/d/image_clustering/image_segmentation/image/test_1001_new.jpg")

