---
author: Isaac
title: Yolo
lang: es
documentclass: paper
date: 16/02/24
font: Arial
fontsize: 12
---
# Yolo Image Classification
There are several kinds of versions
```
YOLOv8n-cls = 4.3 FLOPs 
YOLOv8s-cls = 13.5 FLOPs
YOLOv8m-cls = 42.7 FLOPs
YOLOv8l-cls = 99.7 FLOPs
YOLOv8x-cls = 154.8 FLOPs
```
We can see that in this kind of models there's a letter that is changing first
the n, s, m, l, x. Depending of the letter the algorithim is better but is
heavier and need more resuources to train.

## Supported Datasets

1. Caltech 101 -> images of 101 object categories for image classification tasks.
1. Caltech 256 -> 256 object categories and more challenging images.
1. CIFAR-10 -> 60K 32x32 color images in 10 classes
1. Cifar-100 -> Extended version of CIFAR-10 with 100 object categories and 600 images per class.
1. ImageNet -> A large-scale dataset for object detection and image classification with over 14 million images and 20,000 categories.
1. Your own dataset, base on the same format

[Train-Yolov8](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/train-yolov8-classification-on-custom-dataset.ipynb#scrollTo=jbVjEtPAkz3j)

Flop's = Floating Point Operations Per Second, it's the metrci used to quantify a processor or computing system ability to perform this kind of procedures.

FLOPs are significant for several reasons:
* Computational Efficnecy

* Energy Consumtion

* Model Compairson

* Model Optimization
