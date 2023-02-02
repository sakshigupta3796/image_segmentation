<br/>

# SKU Object Detection and Matching on Planogram Image 

**Objective:** Aim of this Application is to detect SKU(store keeping unit) on a given Planogram image and Cluster/Count the similar objects/products. 

# Architectural Flow
<p align="center">
<img width="834" alt="flow2" src="https://user-images.githubusercontent.com/74641501/215983049-39fbdb35-5c5e-4796-b713-f779570860bf.PNG">
</p>

1. **Object Detection:** In this application user needs to upload an Image that will pass through object detection model that is trained on SKU dataset using YOLOV5. Detected objects/products will be stored for next step(Feature Extraction) with their confidence level and co-ordinates in the image.
2. **Features EXtraction:** At this stage, Image Features Extraction is performed using multiple alogorithms(like VGG16, Image GPT and Image Hashing) on the original and enhanced detected objects. 
3. **Clusturing :** Last step is to implement the clustering algorithms on the extracted features using cosine similarity and hamming distance.

# Explorations
### Possible Approaches at each stage
  <p align="center">
   <img width="834" alt="explore_flow_1" src="https://user-images.githubusercontent.com/74641501/216257737-ded0699d-9cd6-41ab-8b98-8aedfa1ff76c.png">
   </p>
   
1. **Object Detection:** Explored YOLOV5 algorithm for different number of epochs(10 and 20). Achieved best performance with 20 epochs.
2. **Feature Extraction:** Explored different feature extraction algorithms:
    * **RESNET50:** ResNet stands for Residual Network and is a specific type of convolutional neural network (CNN) introduced in the 2015 paper “Deep Residual Learning for Image Recognition” by He Kaiming, Zhang Xiangyu, Ren Shaoqing, and Sun Jian. CNNs are commonly used to power computer vision applications. Please refer this link for more details : https://iq.opengenus.org/resnet50-architecture/
    * **VGG16:** VGG16 proved to be a significant milestone in the quest of mankind to make computers “see” the world. A lot of effort has been put into improving this ability under the discipline of Computer Vision (CV) for a number of decades. VGG16 is one of the significant innovations that paved the way for several innovations that followed in this field. Please refer this link for more details: https://medium.com/@mygreatlearning/everything-you-need-to-know-about-vgg16-7315defb5918

    * **AUTOENCODER:** Autoencoder is a type of neural network where the output layer has the same dimensionality as the input layer. In simpler words, the number of output units in the output layer is equal to the number of input units in the input layer. An autoencoder replicates the data from the input to the output in an unsupervised manner and is therefore sometimes referred to as a replicator neural network. Please refer this link: https://www.mygreatlearning.com/blog/autoencoder/
    
    * **IMAGE GPT:** Please refer this link for details: https://openai.com/blog/image-gpt/
    * **IMAGE HASHING:** This algorithm generally used to find how similar two images are using hashing concept. Please refer this link for details : https://lvngd.com/blog/determining-how-similar-two-images-are-python-perceptual-hashing/
    * **STRUCTURAL SIMILARITY:** Please refer this link: https://medium.com/srm-mic/all-about-structural-similarity-index-ssim-theory-code-in-pytorch-6551b455541e 
    * **Template Matching:** Please refer this link details : https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html 
3. **Clustering:** Applied multiple Clustering algorithms:
    * **K-MEAN CLUSTERING:** K-means clustering is one of the simplest and popular unsupervised machine learning algorithms. Please refer this link for more details : https://www.javatpoint.com/k-means-clustering-algorithm-in-machine-learning 
    * **DBSCAN:** Clustering analysis or simply Clustering is basically an Unsupervised learning method that divides the data points into a number of specific batches or groups, such that the data points in the same groups have similar properties and data points in different groups have different properties in some sense. It comprises many different methods based on differential evolution. 
E.g. K-Means (distance between points), Affinity propagation (graph distance), Mean-shift (distance between points), DBSCAN (distance between nearest points), Gaussian mixtures (Mahalanobis distance to centers), Spectral clustering (graph distance) etc. Fundamentally, all clustering methods use the same approach i.e. first we calculate similarities and then we use it to cluster the data points into groups or batches. Here we will focus on Density-based spatial clustering of applications with noise (DBSCAN) clustering method.  Please refer this link for more details : https://www.analyticsvidhya.com/blog/2020/09/how-dbscan-clustering-works/
    * **COSINE SIMILARITY:** Cosine similarity is a metric, helpful in determining, how similar the data objects are irrespective of their size. We can measure the similarity between two sentences in Python using Cosine Similarity. In cosine similarity, data objects in a dataset are treated as a vector. The formula to find the cosine similarity between two vectors is – Cos(x, y) = x . y / ||x|| * ||y||
    
# Comparission Between Multiple Features Extraction Algorithms: 

  <table>
        <tr>
         <td> <style="font-size:30px"><b>Image Types</b></td>
         <td> <style="font-size:30px"><b>First Image</b></td>
         <td> <style="font-size:30px"><b>Second Image</b></td>
         <td> <style="font-size:30px"><b>Cosine Similarity as per VGG16</b></td>
         <td> <style="font-size:30px"><b>Cosine Similarity as per RESNET50</b></td>
         <td> <style="font-size:30px"><b>Cosine Similarity as per Image GPT</b></td>
         <td> <style="font-size:30px"><b>Hamming Distance(Hashing)</b></td>
         </tr>
        <tr>
          <td>Same Type of Image</td>
          <td> <img src="https://user-images.githubusercontent.com/74641501/216241682-1a9889ab-da67-4859-8722-f05ef92b4c21.PNG" alt="sm1" width = 100px height = 180px> </td>
          <td> <img src="https://user-images.githubusercontent.com/74641501/216241934-77dc6a6c-b259-468d-91b6-3b516b22eb28.PNG" alt="sm2" width = 100px height = 180px> </td>
          <td>0.4300912060694731</td>
          <td> </td>
          <td>0.811369137503929</td> 
         </tr> 
   </table>
      
# Steps Involved 
<p align="center">
<img width="771" alt="steps11" src="https://user-images.githubusercontent.com/74641501/216269200-06e3bf3b-13e1-4e09-a58b-dfbb69361378.PNG">
</p>

# Application Demo

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
