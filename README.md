<br/>

# SKU Object Detection and Matching on Planogram Image 

Objective: Aim of this Application is to detect SKU(store keeping unit) on a given Planogram image and Cluster/Count the similar objects/products. 

# Architectural Flow
<p align="center">
<img width="834" alt="flow2" src="https://user-images.githubusercontent.com/74641501/215983049-39fbdb35-5c5e-4796-b713-f779570860bf.PNG">
</p>

1. Object Detection: In this application user needs to upload an Image that will pass through object detection model that is trained on SKU dataset using YOLOV5. Detected objects/products will be stored for next step(Feature Extraction) with their confidence level and co-ordinates in the image.
2. Features EXtraction: At this stage, Image Features Extraction is performed using multiple alogorithms(like VGG16, Image GPT and Image Hashing) on the original and enhanced detected objects. 
3. Clusturing : Last step is to implement the clustering algorithms on the extracted features using cosine similarity and hamming distance.

# Exploration
1. Object Detection: Explored YOLOV5 algorithm for different number of epochs(10 and 20). Achieved best performance with 20 epochs.
2. Feature Extraction: Explored different feature extraction algorithms:
    * RESNET50: 
    * VGG16:
      * comparission between VGG16 AND RESNET:
    * AUTOENCODER:
    
    * IMAGE GPT:
    * IMAGE HASHING:
    * STRUCTURAL SIMILARITY:
3. Clustering: Applied multiple Clustering algorithms:
    * K-MEAN CLUSTERING:
    * DBSCAN:
    * COSINE SIMILARITY:

# Final Implementation

# Input and Clustured Image
   * Input Image
   ![test_1001](https://user-images.githubusercontent.com/74641501/216232867-0cc57d00-ab77-47cb-8669-e977a0c40731.jpg)
   * Clustered Image
   <img width="577" alt="clustered_img" src="https://user-images.githubusercontent.com/74641501/216233035-a478b5f7-0966-4c9a-8420-3369ef73110f.PNG">



# Limitation

# Future Scope


#
