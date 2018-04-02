# **Traffic Sign Recognition** 



---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./results/flow_line_chart.png "Visualization"
[image2]: ./results/orignal_image.png "Grayscaling"
[image3]: ./results/augment_image.png "Random Noise"

[image4]: ./test_images/3.jpg "Traffic Sign 1"
[image5]: ./test_images/4.jpg "Traffic Sign 1"
[image6]: ./test_images/12.jpg "Traffic Sign 1"
[image7]: ./test_images/13.jpg "Traffic Sign 1"
[image8]: ./test_images/14.jpg "Traffic Sign 1"
[image9]: ./test_images/17.jpg "Traffic Sign 1"
[image10]: ./test_images/25.jpg "Traffic Sign 1"
[image11]: ./test_images/34.jpg "Traffic Sign 1"
[image12]: ./test_images/35.jpg "Traffic Sign 1"


---
### Writeup / README


You're reading it! and here is a link to my [project code](https://github.com/Superxmm/udcity_project2_Traffic-Sign-Classifier)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a flow line chart showing how the data distributed in training set

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale according to the paper that is Traffic Sign 

Recognition with Multi-Scale Convolutional Networks.And I alse tried to use YUV color spaces as inputs,but the 

accuracy is lower than grayscale.


As a last step, I normalized the image data because the scale of features should be controlled in same range.


Here is an example of an original image and an augmented image:

![alt text][image2]![alt text][image3]




#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 GrayScale image   				    | 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x108 	|
| Tanh					|												|
| Local_Response_Normalization|										    |
| Max pooling	      	| 2x2 stride, same padding, outputs 14x14x108 	|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x108 	|
| Tanh					|												|
| Local_Response_Normalization|										    |
| Max pooling	      	| 2x2 stride, same padding, outputs 5x5x108 	|
| Flatten		        | 2700 numbers of units        					|
| Fully connected		| 400 numbers of units    						|
| Tanh					|												|
| Fully connected		| 120 numbers of units    						|
| Tanh					|												|
| Fully connected		| 43 numbers of units    						|
| Softmax				| 43 numbers of classes        					|

 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used AdamOptimizer.And I used the local response normalization to make the non-linear 

modules more sophisticated which has been shown to yield higher accuracy, I set the batch size to 128 with 30 

epochs, and the learning rate was chosen as 0.001.


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 100%
* validation set accuracy of 95.7% 
* test set accuracy of 93.9%


This is the first architecture that was tried,but the numbers of full connect units are different.I only set one 

fully connected layer with 100 units,the accuracy was 0.93 approximately.Apparently,it was underfited.So I add 2 

full connected layers and the numbers of the layers,the accuracy was increased to 0.957,and the validation 

accuray was increased to 0.939.

As to the important design choices,the first important design choice is the architecture,which is learned from  

the article and improved. The next important design choice is the local response normalization,it could make the 

non-linear modules more sophisticated which has been shown to yield higher accuracy.



 

### Test a Model on New Images

#### 1. Choose 9 German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] ![alt text][image7] ![alt text][image8]


![alt text][image9] ![alt text][image10] ![alt text][image11] ![alt text][image12] 


The images were taken from different lighting conditions (saturations, low contrast), motion-blur, occlusions, 

sun glare, physical damage, colors fading and so on.All images were resized with 32*32*3 RGB space.The forth, 

seventh,and tenth image might be difficult to classify because they have dark exposure and complex graphics.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Priority road      		| Priority road   									| 
| Stop     			| Stop 										|
| Road work					| Road work											|
| No entry	      		| No entry					 				|
| Ahead only			| Ahead only      							|
| Turn left ahead			| Turn left ahead      							|
| Speed limit (70km/h)			| Speed limit (70km/h)      							|
| Speed limit (60km/h)			| Speed limit (60km/h)     							|
| Yield			| Yield      							|



The model was able to correctly guess all  traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 93.9%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)


For the first image, the model is relatively sure that this is a Priority road sign (probability of 0.99931), 

and the image does contain a Priority road sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.99931         			| Priority road   									| 
| 0.00027     				| Right-of-way at the next intersection 										|
| 0.00017					| Speed limit (50km/h)									|
| 0.00011	      			| Roundabout mandatory					 				|
| 0.00003				    | Speed limit (100km/h)     							|


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

The characteristics maybe the inner network feature maps react with high activation to the sign's boundary 

outline or to the contrast in the sign's painted symbol.

