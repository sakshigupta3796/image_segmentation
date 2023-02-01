<br/>

# SKU Object Detection and Matching on Planogram Image 

Objective: Aim of this Application is to detect SKU(store keeping unit) on a given Planogram image and Cluster/Count the similar objects/products. 

# Architectural Flow

<img width="834" alt="flow2" src="https://user-images.githubusercontent.com/74641501/215983049-39fbdb35-5c5e-4796-b713-f779570860bf.PNG">


1. Object Detection: In this application user needs to upload an Image that will pass through object detection model that is trained on SKU dataset using YOLOV5. Detected objects/products will be stored for next step(Feature Extraction) with their confidence level and co-ordinates in the image.
2. Features EXtraction: At this stage, Image Features Extraction is performed using multiple alogorithms(like VGG16, Image GPT and Image Hashing) on the original and enhanced detected objects. 
3. Clusturing : Last step is to implement the clustering algorithms on the extracted features using cosine similarity and hamming distance.


# 

Healthy vegetation (chlorophyll) reflects more near-infrared (NIR) and green light compared to other wavelengths. But it absorbs more red and blue light.

This is why our eyes see vegetation as the color green. If you could see near-infrared, then it would be strong for vegetation too. Satellite sensors like Landsat and Sentinel-2 both have the necessary bands with NIR and red.
<p align="center">
  <img src="https://user-images.githubusercontent.com/59726565/176847140-3be272fc-683d-49b7-8783-e8df8d6eff2a.jpg" />
</p>

## Example of NDVI in agriculture

Let&#39;s examine NDVI for an agricultural area with center pivot irrigation. Pivot irrigation rotates on a point creating a circular crop pattern.

In true color, here&#39;s how it looks for red, green, and blue bands. We say true color because it is the same as how our eyes see.
<p align="center">
  <img src="https://user-images.githubusercontent.com/59726565/176847559-083a24a5-f64d-4561-9911-5095ad88e2b3.png" />
</p>


In the formula, you can see how NDVI leverages near-infrared (NIR). So when we put the NIR band to display as red, we get color infrared. We say color infrared because near-infrared is in the red channel. As you can see below, the pivot irrigation vegetation should already be shouting out at you in bright red!
<p align="center">
  <img src="https://user-images.githubusercontent.com/59726565/176847635-fa90a7c3-5546-48bb-b8cb-8aeabdb67895.png" />
</p>

When you apply the formula, bright green indicates high NDVI. Whereas red has low NDVI. So it&#39;s quantifying vegetation by measuring the difference between near-infrared (which vegetation strongly reflects) and red light (which vegetation absorbs).
<p align="center">
  <img src="https://user-images.githubusercontent.com/59726565/176847713-de8de302-30f4-4448-9bc7-d5ec9f5a31fd.png" />
</p>

As you can see, it&#39;s easy to pick out the pivot irrigation circles using NDVI. It makes it easier to classify because of NDVI.


## Pipeline architecture:
<pre>
computation of NDVI and categorizing them from a Landsat8 Image
1. Take a satellite Image or image that has both NIR and Red Bands
2. stack the bands
3. use earthpys normalized_diff function to calculate the NDVI by passing the NIR and Red Bands
4. Plot to visualise cmap for the NDVI thats calculated
5. Create classes and apply to NDVI results
6. MApply the nodata mask to the newly classified NDVI data
7. Define color map
8. Define class names
9. Plot the Data
</pre>
## Requirments

1. Python                 3.9.12
2. earthpy                0.9.4


## Run Locally 

### Create a new python virtual environment and activate it
<pre>
1. Create virtual environment (py -m venv env)
2. Activate virtual environment (click View -> Command palette -> Python: Select interpreter -> enter interpreter path-> Find(Browse your file system to find  a python interpreter)-> find and select the python executable file )
</pre>
### Install dependencies

  ``pip install -r requirements.txt``

## Sample Input & Output 
&emsp;&emsp;&emsp; &emsp; &emsp;&emsp;&emsp;&emsp;&emsp;&emsp; <b>Input</b> &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<b>Output</b><br/>
![image](https://user-images.githubusercontent.com/59726565/176835162-08981cad-cc39-4854-9e2e-bf98c3842835.png)![image](https://user-images.githubusercontent.com/59726565/176834642-150e1ab0-b973-4e7f-b109-45c6f8f645fd.png)

## References:
Would recommend reading the reference links to get a better understanding

1. High level understanding of NDVI <a href="https://gisgeography.com/ndvi-normalized-difference-vegetation-index/">https://gisgeography.com/ndvi-normalized-difference-vegetation-index/</a>
2. NDVI implementation using Earthpy & more applications <a href="https://earthpy.readthedocs.io/en/latest/gallery_vignettes/plot_calculate_classify_ndvi.html"> https://earthpy.readthedocs.io/en/latest/gallery_vignettes/plot_calculate_classify_ndvi.html</a>

Collated by : Phaneendra Mudapaka (Please reach out for any comments or suggestions)
