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
      
      <table>
        <tr>
         <td> <style="font-size:30px">First Image</b></td>
         <td> <style="font-size:30px">Second Image</b></td>
         <td> <style="font-size:30px">Cosine Similarity as per VGG16</b></td>
         <td> <style="font-size:30px">Cosine Similarity as per RESNET50</b></td>
         <td> <style="font-size:30px">Cosine Similarity as per Image GPT</b></td>
         </tr>
        <tr>
          <td> <img src="https://user-images.githubusercontent.com/74641501/216241682-1a9889ab-da67-4859-8722-f05ef92b4c21.PNG" alt="sm1" width = 100px height = 180px> </td>
          <td> <img src="https://user-images.githubusercontent.com/74641501/216241934-77dc6a6c-b259-468d-91b6-3b516b22eb28.PNG" alt="sm2" width = 100px height = 180px> </td>
          <td>0.4300912060694731</td>
          <td> </td>
          <td>0.811369137503929</td> 
         </tr> 
      </table>
      


    * AUTOENCODER:
    
    * IMAGE GPT:
    * IMAGE HASHING: This algorithm generally used to find how similar two images are using hashing concept. Please refer this link for details : https://lvngd.com/blog/determining-how-similar-two-images-are-python-perceptual-hashing/
    * STRUCTURAL SIMILARITY: Please refer this link: https://medium.com/srm-mic/all-about-structural-similarity-index-ssim-theory-code-in-pytorch-6551b455541e
3. Clustering: Applied multiple Clustering algorithms:
    * K-MEAN CLUSTERING:
    * DBSCAN:
    * COSINE SIMILARITY:

# Final Implementation

# Input and Clustured Image

   <table>
        <tr>
         <td> <style="font-size:30px">Input Image</b></td>
         <td> <style="font-size:30px">Clustered Image</b></td>
         </tr>
        <tr>
          <td> <img src=https://user-images.githubusercontent.com/74641501/216232867-0cc57d00-ab77-47cb-8669-e977a0c40731.jpg alt="sm10" width = 480px height = 480px> </td>
          <td> <img src="https://user-images.githubusercontent.com/74641501/216233035-a478b5f7-0966-4c9a-8420-3369ef73110f.PNG" alt="sm20" width = 480px height = 480px> </td>
   </table>

# Limitation
   * Application performance depends on the quality of the input image.
   * Application had hard time in distingues to same colour product.

# Future Scope
   * There are some rooms for the improvement in the performance of this capabilities.
   * Need to try different Image GPT version and observer which version gives best result on our problem. Please refer this link for different vesion of model : https://huggingface.co/models?pipeline_tag=image-classification&sort=downloads&search=google%2Fvit

#
