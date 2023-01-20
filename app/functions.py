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
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.metrics import structural_similarity
import cv2

# !pip install -q git+https://github.com/huggingface/transformers.git datasets
from transformers import ImageGPTFeatureExtractor, ImageGPTModel

# !pip install imagehash
import imagehash

import torchvision
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes
import feature_extractor as fe 

def filter_images(path,t):
  # read text file into pandas DataFrame
  df = pd.read_csv(path, sep=" ",names=["class", "x1","y1","x2","y2","confidence","file_path"])
  df=df[df.confidence>=t]
  df.reset_index(inplace=True,drop=True)
  return df

def filter_cropped(path,path_filter,file_ls):
  os.chdir(path)
  data_GPT= {}
  data_VGG={}
  Filter_list=[ImageFilter.DETAIL,ImageFilter.CONTOUR,ImageFilter.EDGE_ENHANCE,ImageFilter.SHARPEN]
  # creates a ScandirIterator aliased as files
  with os.scandir(path) as files:
    # loops through each file in the directory
      for file in files:
          if (file.name.endswith('.jpg')) and (file.name in file_ls):
              # image = Image.open(path+file.name)
              image=  Image.open(path+file.name)
              image = image.filter(Filter_list[0]())
              enhancer = ImageEnhance.Contrast(image)
              image = enhancer.enhance(2)
              image = image.convert('RGB')
              image.save(path_filter+file.name)

def call_gpt_vgg(path,file_ls,feature_extractor_GPT,model_GPT,device,model_VGG):
  os.chdir(path)
  data_GPT= {}
  data_VGG={}
  # creates a ScandirIterator aliased as files
  with os.scandir(path) as files:
    # loops through each file in the directory
      for file in files:
          if (file.name.endswith('.jpg')) and (file.name in file_ls):
              image = Image.open(path+file.name)
              feat_GPT=fe.extract_features_GPT(image,feature_extractor_GPT,model_GPT,device)
              data_GPT[file.name]=feat_GPT.tolist()
              feat_VGG = fe.extract_features_VGG16(path+file.name,model_VGG)
              data_VGG[file.name] = feat_VGG
          
  print("feature extraction completed")
  return data_GPT,data_VGG

def structural_similarity_fun(image_path,i,j):
    imageA = cv2.imread(image_path+i)
    imageB = cv2.imread(image_path+j)

    # Convert images to grayscale
    before_gray = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    h0, w0 = before_gray.shape
    dim=(w0,h0)
    # dim = before_gray.shape
    after_gray = cv2.resize(after_gray, dim)
    # Compute SSIM between two images
    (score, diff) = structural_similarity(before_gray, after_gray, full=True)
    return score

def clustering_cosine_struture_similarity(data_GPT_org,data_VGG_org,data_GPT_filter,data_VGG_filter,image_path_org,image_path_filter,t1,hashfunc):
    feat_list=list(data_GPT_org.keys())
    # from scipy import spatial
    look_up={}
    for i in feat_list:
      look_up[i]=0
    clusters={}
    k=1

    for i in feat_list:
      if look_up[i]==1:
        pass
      else:
        clusters[k]=list()
        for j in feat_list:
          if i==j or look_up[j]==1:
            pass
          else:
            similarity_GPT_org= -1 * (spatial.distance.cosine(data_GPT_org[i], data_GPT_org[j]) - 1)
            similarity_VGG_org= -1 * (spatial.distance.cosine(data_VGG_org[i], data_VGG_org[j]) - 1)
            similarity_GPT_filter=-1 * (spatial.distance.cosine(data_GPT_filter[i], data_GPT_filter[j]) - 1)
            similarity_VGG_filter=-1 * (spatial.distance.cosine(data_VGG_filter[i], data_VGG_filter[j]) - 1)
            score_org=structural_similarity_fun(image_path_org,i,j)          
            hash10 = hashfunc(Image.open(image_path_org + i))
            hash11 = hashfunc(Image.open(image_path_org + j))
            similarity_hash = hash10 - hash11
            if (similarity_GPT_org>0.65 or similarity_VGG_org>0.7 or similarity_GPT_filter >0.65 or similarity_VGG_filter>0.75 ) and similarity_hash<=7 and score_org>=t1:
              look_up[j]=1
              clusters[k].append((j,similarity_GPT_org,similarity_hash))        
        clusters[k].append((i,1,1))
        look_up[i]=1
        k=k+1
    return clusters

def plot_clusters(clusters,image_path):
    keys=list(clusters.keys())
    for key in keys:
      for img in clusters[key]:
        print(key)
        print(img[0],img[1],img[2])
        img = mpimg.imread(image_path+img[0])
        imgplot = plt.imshow(img)
        plt.show()

def images_after_clustering(img_path,image_org, data,groups):
  # bounding box in (xmin, ymin, xmax, ymax) format
  # top-left point=(xmin, ymin), bottom-right point = (xmax, ymax)
  for key in groups.keys():
    print("cluster:", key)
    color = (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))
    for img22 in groups[key]:
      print("inside:")
      print("img22", img22[0])
      x1 = data[data["img_name"]==img22[0]]['x1'].values[0]
      y1= data[data["img_name"]==img22[0]]['y1'].values[0]
      x2 = data[data["img_name"]==img22[0]]['x2'].values[0]
      y2 = data[data["img_name"]==img22[0]]['y2'].values[0]
      bbox = [int(x1), int(y1), int(x2), int(y2)]
      bbox = torch.tensor(bbox, dtype=torch.int)
      bbox = bbox.unsqueeze(0)
      # draw bounding box on the input image
      image_org =draw_bounding_boxes(image_org, bbox, width=10, colors=color)

  # transform it to PIL image and display
  image_org = torchvision.transforms.ToPILImage()(image_org)
  return image_org


  model_name()
