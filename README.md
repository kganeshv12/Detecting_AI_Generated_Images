# CNN vs ViT for AI and Real Image Classification

## Overview
This repository contains the implementation and comparison of Convolutional Neural Networks (CNN) and Vision Transformers (ViT) for the classification of AI and real data. The goal is to explore the performance differences between these two popular architectures in the field of computer vision.

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Models](#models)
4. [Results](#results)
5. [Conclusion](#conclusion)


## Introduction
Convolutional Neural Networks (CNN) have been the backbone of many computer vision tasks for years. However, Vision Transformers (ViT) have gained attention for their ability to handle image classification tasks without relying on convolutional layers. This project aims to compare the performance of CNN and ViT architectures on a dataset containing AI and real images.

## Dataset
The dataset we have used is **CIFAKE: Real and AI-Generated Synthetic Images**

The dataset contains two classes - REAL and FAKE.

For REAL, images are from Krizhevsky & Hinton's CIFAR-10 dataset

For the FAKE images, are generated from the equivalent of CIFAR-10 with Stable Diffusion version 1.4

There are 100,000 images for training (50k per class) and 20,000 for testing (10k per class)

CIFAKE is a dataset that contains 60,000 synthetically-generated images and 60,000 real images (collected from CIFAR-10). 

## Models
Explain the architectures of both CNN and ViT models used in the project. Include details such as the number of layers, parameters, and any modifications made to the original architectures.

| Model Name                   | 
|------------------------------|
| CNN_EfficientNetB4   | 
| CNN_EfficientNetB5.ipynb    |
| CNN_EfficientNetB6.ipynb    |
| CNN_InceptionV3.ipynb       |
| CNN_MobileNetV2.ipynb       |
| CNN_MobileNetV3.ipynb       |
| CNN_ResNet101V2.ipynb       |
| CNN_VGG16.ipynb             |
| vit-amunchet-rorshark-vit-base.ipynb |
| vit-base-patch16-224.ipynb  |     
| vit-base-patch32-384.ipynb  |    
| vit-google-vit-base-patch16-224-in21k.ipynb |  
| vit-tiny-patch16-224.ipynb  |     
| vit_dima806.ipynb           |

## Results
We present the results of the model comparison, including accuracy, precision & recall.

| Model Name                   | Accuracy | Precision | Recall | Parameters |
|------------------------------|----------|-----------|--------|------------|
| CNN_EfficientNetB4.ipynb    |     | -          | -      |    |
| CNN_EfficientNetB5.ipynb    |      | -          | -      |     |
| CNN_EfficientNetB6.ipynb    |      | -          | -      |     |
| CNN_InceptionV3.ipynb       |   | -          | -      |  |
| CNN_MobileNetV2.ipynb       | | -          | -      |     |
| CNN_MobileNetV3.ipynb       | | -          | -      |     |
| CNN_ResNet101V2.ipynb       | | -          | -      | |
| CNN_VGG16.ipynb             |   | -          | -      |     |
| vit-amunchet-rorshark-vit-base.ipynb |  | - | - |  |
| vit-base-patch16-224.ipynb  |     | -          | -      |  |
| vit-base-patch32-384.ipynb  |    | -          | -      |  |
| vit-google-vit-base-patch16-224-in21k.ipynb |  | - | - |  |
| vit-tiny-patch16-224.ipynb  |     | -          | -      |  |
| vit_dima806.ipynb           | -            | -          | -      | -              |


## Conclusion

From our experiments we can conclude that:

The best performing model is : 

and on an average CNNs perform better than ViTs


![Sample Image](comparison_chart.jpg)
