# Mixed _ Models :The aim of this project is to put into practice some clustering methods and to evaluate them using appropriate metrics.

# Datasets used : JAFFE, MNIST5, MFEA, USPS, OPTIDIGITS

library(R.matlab)
library(FactoMineR)
library(ggplot2)
library(ggfortify)
library(factoextra)
library(NbClust)

## Data Preparation 

jaffe_source <- readMat("/home/hajji/Desktop/Mixed Models/Data-partie-1/DATA_MATLAB - Projet-master-MLDS/jaffe.mat")


jaffe_x = data.frame(jaffe_source[["X"]])



jaffe_y = data.frame(jaffe_source[["y"]])

jaffe_y = data.frame(t(jaffe_y))

rownames(jaffe_y) <- 1:nrow(jaffe_y)

## PCA 

x.pca <- PCA(jaffe_x, scale.unit=TRUE, graph=T)

plot(x.pca)

## NBCLUST 

# Kmeans

x.kmeans <- NbClust(data = jaffe_x, diss = NULL, distance = "euclidean", min.nc = 2, max.nc = 10, method = "kmeans",index = "silhouette")

x.average <- NbClust(data = jaffe_x, diss = NULL, distance = "euclidean", min.nc = 2, max.nc = 10, method = "average",index = "silhouette")

x.ward <- NbClust(data = jaffe_x, diss = NULL, distance = "euclidean", min.nc = 2, max.nc = 10, method = "ward.D",index = "silhouette")

x.single <- NbClust(data = jaffe_x, diss = NULL, distance = "euclidean", min.nc = 2, max.nc = 10, method = "single",index = "silhouette")

x.comp <- NbClust(data = jaffe_x, diss = NULL, distance = "euclidean", min.nc = 2, max.nc = 10, method = "complete",index = "silhouette")

plot(x.kmeans$Best.partition)
plot(x.average$Best.partition)
plot(x.comp$Best.partition)
plot(x.single$Best.partition)
plot(x.ward$Best.partition)

x.HCPC <- HCPC(x.pca, graph = FALSE, max = 10)

plot(x.HCPC)
plot(x.HCPC)




