---
layout: post
title: DTW- Dynamic Time Wrapping
image: /img/Blog/img/dtw.png
tags: [Temporal Analysis, Techniques, Statistics]
use_math: true
---

DTW is a family of algorithms which compute the local stretch or compression to apply to the time axes of two time series in order to optimally map one  onto the other.  DTW outputs the remaining cumulative distance between the two and, if desired, the mapping itself (warping function).  In another word they can compute the similarity between time series which may vary in time (i.e. wrap in time). 

DTW is widely used e.g. for classification and clustering tasks in Econometrics, Chemometrics and general time series mining. Originally designed for speech recognition, Basically DTW finds the optical global alignment between two time series by exploiting temporal distortion between them.

Multiple studies as confirmed that for the problem involving time series data for example a classification task, algorithm exploiting the DTW to wrap the invariance among the signal is hard to beat. [1]\[2]

##### Things to be considered while implementing or using DTW:

1. Z Normalizing
2. just-in-time normalization
3. early abandoning techniques
4. Applying endpoint constrain
5. setting up wrapping window  

##### Computing DTW, 

Compute the distance matrix(cost matrix) between every possible pair of points between the two time series. Any possible wrapping between two time series will be a path through the computed cost matrix. 

$DTW(q,c) = min \Bigg( \frac{\sqrt{\sum_{k=1}^{K}w_k)}}{K} \Bigg)$, here w is the wrapping constant.

The optimal (minimum) path or wrapping between two time series will provide us with the DTW, which can be obtained using the below recursive function.

$\gamma(i,j) = distance(q_i,c_i) + min( \gamma(i-1,j-1), \gamma(i-1,j), \gamma(1,j-1)) $



Its important to note that DTW is symmetric and has constancy of self similarity but does not follow positivity (separation)  and triangular inequality.  This would be the best way to say DTW is a distance measure and not a metric. Now interesting if we increase the sample space to an enormous level, we would be able to find lots of A,B, and C which would follow triangular inequality(just read this over the internet try to find the source). Mathematically in the limit, as w tends to zero DTW is a metric, interesting isn't it?

##### Generalizing DTW to Multidimensional Data:

This can be achieved in two way, independent (calculate DTW separately for each dimension and sum up) and dependent DTW computation



#### Reference

[1]  Bagnall, A., Lines, J., Bostrom, A., Large, J., & Keogh, E. (2017). The great time series classification bake off: a review and experimental evaluation of recent algorithmic advances. *Data Mining and Knowledge Discovery*, *31*(3), 606-660.

[2]  Bagnall, A., Lines, J., Bostrom, A., Large, J., & Keogh, E. (2017). The great time series classification bake off: a review and experimental evaluation of recent algorithmic advances. *Data Mining and Knowledge Discovery*, *31*(3), 606-660.





