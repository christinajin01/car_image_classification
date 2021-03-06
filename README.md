# Densenet for Fine-Grained Vehicle Image Classification
*Christina Jin*\
*May, 2020*

## Problem Statement: 
This project aims to classify car model names from [the Comprehensive Cars (CompCars) dataset](http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/index.html). This dataset contains car image data from two scenarios, web-nature and surveillance-nature. Since the purpose of the task aims for higher accuracy, I've decided to use the web-nature data due to its higher resolution and potential of capturing greater details. The web dataset, in particular, contains 86,726 car images and 1,716 car models. 

## Approaches:
Due to limitation of computing power, I've decided to train 1000 images with 18 different car models and test on 500 images with 10 car models.

![](https://github.com/christinajin01/car_image_classification/blob/master/plots/resnet_densenet_arch.png)
Incentivized by this paper by Valev et al. on [A Systematic Evaluation of Recent Deep Learning Architectures for Fine-Grained Vehicle Classification](https://arxiv.org/abs/1806.02987), I decided to implement Densenet 121 as my base model. Densenet builds on **ResNet** (*left image above*), which tries to mitigate the problem of vanishing gradient by adding an identity shortcut to the next layer. The addition of such shortcut performs elementwise addition and allows gradient to skip over intermediate layers and backprogate without being largely diminished. **DenseNet** (*right image above*) follows such idea by densely connecting every layer to every other layer inside a block, and replacing the elementwise addition with concatenation to preserve information. This is particularly important for image classification tasks like this, because we don't want to lose too many image details over the course of the whole CNN network. Such concatenation also reduces the number of parameters needed to train and largely improves the computation complexity. I chose the best performing DenseNet model available on Keras (**DenseNet 121**) and used transfer learning with pre-trained ImageNet weights. 

### 1. Base Model:
My first attempt was to use DenseNet 121 model directly with pre-trained ImageNet weights. 
For data preprocessing, I squared the images with 0 paddings and resized them down to equal sizes. I then added a fully connected layer with ReLU activation and dropout. 

| Layers  | 
|:-------------:|
| DenseNet121 |
| Flatten | 
| Dense |
| Batch Normalization |
| Activation |
| Dropout |
| Dense |

Accuracy                   |  Loss
:-------------------------:|:-------------------------:
![](https://github.com/christinajin01/car_image_classification/blob/master/plots/base_mod_acc.png)  |  ![](https://github.com/christinajin01/car_image_classification/blob/master/plots/base_mod_loss.png)

Training accuracy: 1.000\
Validation accuracy: 0.775\
Testing accuracy: 0.782

### 2. Data Augmentation:
To improve the base model, I augmentated the data to include rotated and horizontally flipped image, since our model should only be focused on features like shapes, each part's relative position, colors etc., instead of its absolution position in the image. 

Accuracy                   |  Loss
:-------------------------:|:-------------------------:
![](https://github.com/christinajin01/car_image_classification/blob/master/plots/aug_mod_acc.png)  |  ![](https://github.com/christinajin01/car_image_classification/blob/master/plots/aug_mod_loss.png)

Training accuracy: 1.000\
Validation accuracy: 0.790\
Testing accuracy: 0.786

### 3. Image Cropping:
After exploring the misclassified car images, I found that car images with a relatively clear background tend to have higher classification accuracy than those with more background noise. Thus, I tried cropping the images based on the bounding boxes given in the dataset. (Note: There are multiple available bounding box algorithms, including YOLO.) The images are cropped and padded, and then fed into the same network as described above (with data augmentation). 

Accuracy                   |  Loss
:-------------------------:|:-------------------------:
![](https://github.com/christinajin01/car_image_classification/blob/master/plots/final_mod_acc.png)  |  ![](https://github.com/christinajin01/car_image_classification/blob/master/plots/final_mod_loss.png)

Training accuracy: 1.000\
Validation accuracy: 0.810\
Testing accuracy: 0.836

## Results: 
To see a detailed breakdown of how well the model performed, I plotted the confusion matrix below:

![](https://github.com/christinajin01/car_image_classification/blob/master/plots/confusion_matrix.png)

As seen from the matrix plot, Audi Q3 has the lowest recall, whereas Audi Q5 has the highest. 

I've trained this model on a larger dataset with 5,000 train images and 73 labels, with accuracy and confusion matrix reported below:\
Training accuracy: 0.9995\
Validation accuracy: 0.813\
Testing accuracy: 0.827

![](https://github.com/christinajin01/car_image_classification/blob/master/plots/cm_5000.png)

We see that the current architecture is able to perform decently well even with a larger variety of labels. 

## Limitations & Future work:
### 1. Size of dataset:
The current approach was limited by computing power. Although it performed well on the selected labels, further testing is needed for a greater variety of car models/types of pictures taken to see how well such model generalizes. 

### 2. Bounding box and background removal:
A bounding box generation pipeline is still needed for future images. One of the most popular framework is YOLO (You Only Look Once), which performs object detection and outputs the bounding box coordinates of the object. Moreover, a background removal (e.g.: Unet) rather than just the bounding box might work even better in removing unnecessary information encoded by the images. 

### 3. Shadows and glares:
It's brought to attention that certain images with shadows and glares are relatively harder to classify because such noise contaminates the object directly and is potentially impossible to eliminate just by background removal. To deal with such problem, we can try doing data augmentation by generating similar images using GANs.

## What have I learned:
In this project, I got to experiment with some popular techniques for image maniputation as well as popular architecture for fine-grained image classification. From reading research papers, implementing architectures, to finding bugs and waiting for models to finish training, I've learned that such task takes an enormous amount of researching and patience. This project has also honed my skills in Keras and I'm excited to delve more into the field of computer vision. 
