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


import torchvision
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes

def extract_features_GPT(image,feature_extractor_GPT,model_GPT,device):
  encoding = feature_extractor_GPT(images=image, return_tensors="pt")
  pixel_values = encoding.to(device)
  # pixel_values = encoding.pixel_values.to(device)
  # forward through model to get hidden states
  with torch.no_grad():
    outputs = model_GPT(**pixel_values, output_hidden_states=True)
  hidden_states = outputs.hidden_states
  features = hidden_states[-1].reshape(-1,524288)
  # features = torch.mean(hidden_states[-1], dim=1)
  feat_dict  = features.cpu().detach().numpy()
  
  return feat_dict

def extract_features_VGG16(file, model):
    # load the image as a 224x224 array
    img = load_img(file, target_size=(224,224))
    # convert from 'PIL.Image.Image' to numpy array
    img = np.array(img) 
    # reshape the data for the model reshape(num_of_samples, dim 1, dim 2, channels)
    reshaped_img = img.reshape(1,224,224,3) 
    # prepare image for model
    imgx = preprocess_input(reshaped_img)
    # get the feature vector
    features = model.predict(imgx, use_multiprocessing=True)
    print("features",features)
    return features

