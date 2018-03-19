---
layout: post
title: Balancing techniques for unbalanced datasets 
subtitle: in Python & R
---

**Data Sampling** in data science is an important aspect for any statistical analysis project which is used to select, manipulate and analyze a representative subset of data points called samples in order to identify patterns and trends in the larger data set usually termed as population being examined. Lets say we are conducting a survey in the United States, and we collected data from 1000 samples. The information from these samples can then be used to infer details about the entire population of the United States. The essential point to be noted here is that these sample points should in turn be very close to the population we are examining. If the data is normally distributed, then the samples can be randomly chosen. However if we are dealing with unbalanced dataset where number of instances of one class predominates the other, then we must use balancing strategies for selecting random samples to avoid majority class bias.

I have already written a blog post indicating the use of stratified sampling – Undersampling and Oversampling and their use, link is Data Splitting in machine learning for high variance dataset.

In this post, I am writing Undersampling and Oversampling methods that can be implemented to overcome the problem that exist with unbalanced dataset. Depending on the data science problem, one can decide which method to implement.

**Undersampling Methods**: It reduces the number of observations from majority class to balance the data set. The various methods of undersampling implemented in this notebook includes:
- UnderSampler: Randomly under-samples the majority class with replacement
- TomekLinks: Identifies all Tomek links between the majority and minority classes
- ClusterCentroids: Under-sampling with Cluster Centroids using K-means
- NearMiss method: Selects the majority class samples which are close to some minority class samples
- Condensed Nearest Neighbour: Selects subset of instances that are able to correctly classifying the original datasets using one-nearest neighbor rule.
- One Side Selection: Method resulting from the application of Tomek links followed by Condensed Nearest Neighbor.
- Neighborhood Cleaning Rule: Utilizes the one-sided selection principle, but considers more carefully the quality of the data to be removed.

**Oversampling Methods**: Replicates the observations from minority class to balance the data. The various methods of oversampling implemented in this notebook includes:
- RandomOverSampler: Randomly over-samples the minority class with replacement.
SMOTE- Synthetic Minority Over-sampling Technique: It works by creating synthetic samples from the minor class instead of creating copies.
bSMOTE- Borderline SMOTE: Minority samples near the borderline are over-sampled.
SVM_SMOTE- Support Vectors SMOTE: The SVM smote model fits a support vector machine classifier to the data and uses the support vector to provide a notion of boundary. Unlike regular smote, where such notion relies on proportion of nearest neighbours belonging to each class.
SMOTE + Tomek links (Combines Over-sampling followed by under-sampling): Performs over-sampling using SMOTE and cleaning using Tomek links.
SMOTE + ENN (Combines Over-sampling followed by under-sampling): Performs over-sampling using SMOTE and cleaning using Edited Nearest Neighbours (ENN).
EasyEnsemble: Create an ensemble of balanced sets by iteratively under-sampling the imbalanced dataset using an estimator.
BalanceCascade: BalanceCascade is similar to EasyEnsemble except that it removes correctly classified major class examples of trained learners from further consideration.
The IPython Notebook implementation of the above methods can be found on Github Link.

To learn more about the Imbalance Learn API in Python that I used to implement the methods please refer to the documentation: http://contrib.scikit-learn.org/imbalanced-learn/api.html

For R implementation of above methods refer to Package ‘unbalanced’: https://cran.r-project.org/web/packages/unbalanced/unbalanced.pdf
