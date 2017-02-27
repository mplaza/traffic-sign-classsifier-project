#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/mplaza/traffic-sign-classsifier-project/blob/master/Traffic_Sign_Classifier.ipynb) or in [html format](https://github.com/mplaza/traffic-sign-classsifier-project/blob/master/report.html)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the python and numpy to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third and fourth code cell of the IPython notebook.  
In the fourth cell I plotted a random sampling of the images with their associated labels to better understand the content of the dataset. 

I also calculcated the number of examples in each class in the training dataset and visualized the distribution with a boxplot

![boxplot](/writeup_images/boxplot.png)

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.


I experimented with converting to YUV since this is more similar to human perception, but decided to convert the images to grayscale to reduce the image dimensions and make the model easier to train. I considered using hisogram equalization to improve contrast, but since only the traffic sign and not the image background are imported, decided to use CLAHE to consider local rather than global image contrast. (Cell 8)


![grayscale](/writeup_images/grayscale.png)

Then I normalized the image data for better conditioning. The cv2 grayscale conversion changed the shape of the image so I reshaped it to make it compatible with the lenet architecture that we used previously. (Cell 10)

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

In the fifth cell of the IPython notebook, I added additional images to the classes that had many fewer images in the training dataset. I calculated the number of images that each class should have as no less than .5 standard deviations less than the mean in order to make the distribution more regular. This was done to prevent the network from being biased towards the larger classes and unable to classify these smaller classes correctly.

I also added 2 additional sets of jittered images for each image in the dataset to make the learning more robust to these types of pertubations (I used scaling and rotation) and reduce overfitting my adding random noise. This was shown to decrease validation error in the Sermanet & LeCun paper and seemed to decrease validation error in my model as well. (Cell 6)

![jittered](/writeup_images/jittered.png)

I shuffled the data and used train_test_split to set aside 20% of the data for validation, leaving the testing data to test against after the model was finetuned. (Cell 13, 14)

My final training set had 94173 number of images. My validation set and test set had 23544 and 12630 number of images.


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the sixteenth cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   					| 
| Convolution 5x5     	| 1x1 stride, SAME padding, outputs 32x32x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x6 					|
| Convolution 5x5	    | 1x1 stride, VALID padding, outputs 12X12x6	|
| RELU					| 	        									|
| Avg pooling			| 2x2 stride,  outputs 6X6X16					|
| Flatten				| output 576									|
| Fully Connected 		| output 120									|
| RELU 					| 												|
| Fully Connected 		| output 84										|
| RELU					| 												|
| Dropout				| keep prob 0.7									|
| Fully Connected 		| output 10										|


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the cell 18-20 of the ipython notebook. 

To train the model, I used an adam gradient descent optimizer to minize loss.

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in cells 20 and 21 of the Ipython notebook.

My final model results were:
* training set accuracy of 98.2%
* validation set accuracy of 96.9% 
* test set accuracy of 94.1%

I began with the lenet architecture from the previous lab because lenet architectures are well suited to these types of image processing problems and it seemed to be a better approach to fine tune the model than start from scratch. Initially the model performed quite well and most of the modifications I made hurt overall performance. I tried removing a convolution layer and it seemed to be underfitting the data. Changing the shapes of the convolution layers didn't seem to help accuracy. I noticed that the model began to overfit after a certain point, and introducing a dropout layer and adding more jittered data to the dataset improved validation accuracy. I experimented with removing pooling layers after adding dropout, but the results were not better and training time was increased. I also noticed that after the eigth epoch, the validation accuracy would often decrease and rarely were there any major gains after this, so I applied early termination and decreased the epochs to 8 to avoid overfitting.

The good performance of the Lenet architecture on this model makes sense given the similarity of traffic sign classification to character recognition. The combination of convolution, pooling and activation allows enough feature extraction that the fully connected layers can use for classification. Other architectures like AlexNet, GoogleNet, or ResNet might perform even better on a task like this, but Lenet was a good starting point for this problem.
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text](/sample_german_traffic_signs/3.jpg)  ![alt text](/sample_german_traffic_signs/4.png) ![alt text](/sample_german_traffic_signs/28.jpg) ![alt text](/sample_german_traffic_signs/40.jpg) 
![alt text](/sample_german_traffic_signs/27.jpg)  

The first image might be difficult to classify because it is slightly distored from processing to 32x32. It looks like the dimensions weren't preserved and this type of transformation wasn't included in the training set so the model might have difficulty classifying it.

The second image has a sharp contract between the hill and sky in the background which might make it more difficult to classify because during image processing this wouldn't be removed and this would represent a sharp contrast line that the model might try to use to classify the image but isn't part of the traffic sign.

The next two signs might be difficult to classify becuase they are quite similar except for the people on the inside, which when the dimensions are reduced become quite similar looking. The child crossing sign might be particularly difficult because there is a kindergarten sign right underneath it, which the model did not train on and so it might classify this as a different type of sign that includes a square shape.

The final sign might be difficult to classify because like the first sign, it appears to be a bit distorted. This also might be diffult because this was one of the classes that started out with less data in the training data set.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the 24th cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Pedestrians      		| Right-of-way at the next intersection   		| 
| Children Crossing    	| Bicycles crossing								|
| 60 km/h				| Roundabout									|
| 70 km/h	      		| 70 km/h					 					|
| Roundabout			| Roundabout	      							|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 40%. This is much less than the accuracy on the test set. Although I didn't modify the model after running it on the test set (so I wasn't training with the test set), it seems that the data in the original dataset had characterisics that the image I got off the web didn't and my model had trouble generalizing. 

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 25th cell of the Ipython notebook.

For the last two images, the model is relatively sure that they are roundabout and 70 km/h signs (probability of .98 and .99), and the predictions are correct. 

The first three images were wrong. The first image has a .68 probability of being a right of way sign, but is a pedestrians sign. The next two most probable are a children crossing and then the correct pedestrian crossing. 

The second sign had .91 probability of being a bicycles crossing but was a children crossing. The children crossing was the next most likely. It seems for the crossing signs, my suspicion that the model might not be able to distinguish between the different crossing signs with new images was correct. More of these types of signs should be loaded into the initial dataset or image processing that might enable the model to distinguish between the inner contents of the signs might be helpful. Maybe the pooling layers hurt with this.

The 60km/h sign was incorrectly labeled as a roundabout with .88 probability and the 60 km/h option wasn't even in the top 5, so the model did a pretty bad job with this one.



| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .68         			| Right-of-way at the next intersection			| 
| .91     				| Bicycles crossing								|
| .88					| Roundabout									|
| .98	      			| 70 km/h						 				|
| .99				    | Roundabout 	     							|

