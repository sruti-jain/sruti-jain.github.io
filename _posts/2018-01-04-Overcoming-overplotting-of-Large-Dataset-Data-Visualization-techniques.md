---
layout: post
title: First post!
image: /img/hello_world.jpeg
---

Data visualization is an important aspect of any data analysis. Representation of information in the form of charts, diagram or picture provides valuable insight about the data including outliers, hot spots and missing data. Especially with large dataset where there are simply too many data points to examine each one, visualizations are helpful tools for understanding the properties of dataset. Having said that usually plotting of large dataset leads to a problem commonly called as overplotting.

Occlusion of data by other data is called overplotting, and it occurs whenever a datapoint or curve is plotted on top of another datapoint or curve, obscuring it. It’s thus a problem for scatterplots, curve plots, 3D surface plots, 3D bar graphs, and any other plot type where data can be obscured. First lets visualize what is Overplotting using Holoviews.

Overplotting

The following plotting techniques discussed can thus be use to overcome overplotting of Large Scale Data.

1 Data Shadowing and Data Shader:

Plotting millions of data point on a Scatter plot directly is not possible with occlusion, therefore we can use the techniques of Data shadowing and shading for plotting large scale data interactively. Data shading automatically reveals valuable details about the data like outliers and missing data.

Below is an example of Data shading and Data shadowing plots of data of 4 million new york taxi trips. The python notebook for the below data visualization is on my Github.

Shading

2 Annotated heatmaps & interactive plots: 

A heatmap is a pictorial representation of data where individual points are contained in a matrix that is represented as colors. A simple heat map provides an immediate visual summary of information. More elaborate heat maps allow the viewer to understand complex data sets. The methods discussed until now can be used for Explanatory visualizations for telling data stories to indicate key findings. But at times, we want the user to interactively explore the data, to unearth their own understanding. In that case, we need to build Exploratory visualizations. These visualizations also play an important role to indicate the variation of one attribute with respect to another and therefore are powerful visualization tools especially in case of large datasets.

A heatmap or annotated heatmap can be build using Seaborn library in Python which is build on top of matplotlib for statistical visualization that aims at summarizing data yet showing the distribution of data. Interactive plots can be constructed using Bokeh library that targets modern web browsers for presentation. Its goal is to provide elegant, concise construction of novel graphics in the style of D3.js, and to extend this capability with high-performance interactivity over very large or streaming datasets. Another Library widely used for interactive plots is Holoviews  that makes analyzing and visualizing scientific or engineering data much simpler, more intuitive, and more easily reproducible. Example visualization plots using these libraries are uploaded on my Github. Below are a few images of the visualizations I created.

Holoviews

By now, you must have realized, how data can be presented using visualization to unearth useful insights. I find performing visualization in Python much easier as compared to R. In this article, I have discussed about deriving various visualizations in Python for Large Datasets. In this process, I made use of matplotlib, seaborn, holoviews & Bokeh libraries in python.
