# AdvancedMachineLearning-Project
This repository contains reports and codes for Advanced Machine Learning course project.

# Group Members:
Pinar Goktepe, 
Dusan Mihajlov, 
Michael Baur

# Project Description
Deep neural networks need many labeled data instances to be able to provide significant results. However in many cases the number of labeled data is limited. Therefore, several methods are developed to handle labeled data availability problem. One way is to use a large labeled dataset such as ImagNet, COCO or Pascal VOC to pretrain the network and then fine tune it by target dataset. Another widely used method is upsampling meaning that increasing number of labeled data instances by shifthing, rotating or other simple data manipulation techniques. 

While labeled data has great amount of cost, unlabeled data comes for free in most cases. The main idea of self supervised learning is to make use of unlabeled data which is easy to gather. Therefore self-supervise learning is developed by benefitting from simple tasks on unlabeled images. Without pretraining, a standard network training starts by random weight initialization. However in self supervised learning, initialization will be done by using a proxy task such as solving jigsaw puzzle, colorization, inpainting etc. rather than a random start. While a network is trying to solve a proxy task on unlabeled data, it learns about data itself. So, the extracted features from this network can be used for starting point of training of another network aiming to classify labeled data instances of the same dataset.

The aim of this project is to classify clothes images via self supervised learning. A good candidate dataset for this task would be DeepFashion database in which 289,222 clothes images are provided. The database contains not only the shop images which are well posed but also customer images. With 50 categories of diverse content, this database will be suitable for goal of this project which is experimenting self supervised learning methods.  

