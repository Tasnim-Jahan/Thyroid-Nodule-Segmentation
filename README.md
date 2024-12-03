# Thyroid-Nodule-Segmentation

**Comparative Analysis of U-Net and Pretrained Encoder Models for Image Segmentation**
This repository contains a complete pipeline for image segmentation using the U-Net architecture and its variations with pretrained backbones such as VGG16, VGG19, ResNet50, and MobileNetV2. The project focuses on evaluating the performance of these models on a medical imaging dataset, leveraging deep learning for precise segmentation tasks.

**Overview**
Image segmentation is a critical task in computer vision, particularly in medical imaging, where it aids in diagnosis and treatment planning. This project aims to:

1. Implement and compare U-Net and U-Net variations with pretrained backbones.
2. Evaluate models using metrics like Dice Coefficient, IoU, Precision, Recall, and F1 Score.
3. Visualize training progress and results to assess model performance effectively.

**Features**
* Data Preprocessing and Augmentation: Uses albumentations for advanced image and mask augmentation.

* Model Architectures:
  
   ** Baseline U-Net
  
   ** U-Net with VGG16, VGG19, ResNet50, and MobileNetV2 as encoders.

* Custom Loss Functions:

  ** Dice Loss

  ** IoU as an evaluation metric.

*Performance Metrics:
   
   ** Dice Coefficient, Accuracy, IoU, Precision, Recall, and F1 Score.

* Visualization:

  ** Training and validation loss and metric progression over epochs.

  ** Comparative bar charts and line plots of metrics across models.

* Model Evaluation:

  ** Quantitative analysis using test metrics.

  ** Qualitative analysis through visualized segmentation predictions.

**Dataset**
The project uses a medical imaging dataset structured into training, validation, and test sets. The dataset must contain:
   
   ** Images in directories like trainval-image and test-image.
   
   ** Corresponding masks in trainval-mask and test-mask.

**Methodology**
1. Data Preprocessing:

   ** Images and masks are resized to (256, 256).

   ** Data augmentation using albumentations.

3. Models:
   ** Implemented U-Net as the baseline.

   ** Extended U-Net with pretrained backbones:


   1. VGG16
   2. VGG19
   3. ResNet50
   4. MobileNetV2

5. Training:

   ** Optimizer: Adam with a learning rate of 1e-4.

   ** Loss Function: Dice Loss.

   ** Early stopping and learning rate reduction callbacks.

7. Evaluation:

   ** Metrics: Dice Coefficient, IoU, Accuracy, F1 Score, Precision, and Recall.

   ** Visual comparison of segmentation results.
   
**Requirements**
To replicate this project, install the following dependencies:
   
   ** Python 3.7+
   
   ** TensorFlow 2.x
   
   ** OpenCV
  
   ** Albumentations
   
   ** Pandas
   
   ** NumPy
   
   ** Matplotlib
   
   ** Seaborn


**Quantitative Evaluation:**

Model	Dice Coefficient	IoU	Precision	Recall	Accuracy

U-Net	0.85	0.78	0.88	0.86	0.90

VGG16 U-Net	0.87	0.80	0.89	0.88	0.92

VGG19 U-Net	0.88	0.82	0.91	0.89	0.93

ResNet50 U-Net	0.86	0.79	0.89	0.87	0.91

MobileNetV2	0.84	0.76	0.87	0.85	0.89

**Qualitative Evaluation:**

Visualizations of segmentation results are available in the results/ directory for each model.

**Plots**

* Training Progress:

  ** Loss and metric progression for training and validation.

* Model Comparison:

  ** Bar charts comparing metrics across models.

  ** Line plots for training and validation metrics.

**Acknowledgments**

* U-Net architecture by Ronneberger et al.

* Pretrained models from TensorFlow's Keras Applications.

