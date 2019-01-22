## Final version : Mixed Models 22/01/2019 at 21:30
# Mixed _ Models :The aim of this project is to put into practice some clustering methods and to evaluate them using appropriate metrics.

# Datasets used : JAFFE, MNIST5, MFEA, USPS, OPTIDIGITS

library(R.matlab)
library(FactoMineR)
library(ggplot2)
library(ggfortify)
library(factoextra)
library(NbClust)
library(clues)
library(Rtsne)
library(aricode)
library(caret)
library(mclust)
library(clues)
library(Rtsne)
library(keras)


## Data Preparation

#################################################
# Importing data from Matlab .mat formatted files
#################################################


jaffe = readMat("Bureau/jaffe.mat")
jaffeData = as.data.frame(cbind(jaffe$X, t(jaffe$y)))

mfea = readMat("Bureau/MFEAT1.mat")
mfeaData = as.data.frame(cbind(mfea$X, mfea$y))

mnist = readMat("Bureau/MNIST5.mat")
mnistData = as.data.frame(cbind(mnist$X, mnist$y))

optidigits = readMat("Bureau/Optdigits.mat")
optidigitsData = as.data.frame(cbind(optidigits$X, optidigits$y))

usps = readMat("Bureau/USPS.mat")
uspsData = as.data.frame(cbind(usps$X, usps$y))


########################################
# Applying PCA to have a visual on Data
########################################

jaffe.res.pca = PCA(jaffeData[,-677], scale.unit = TRUE, ncp = 2)
jpeg("jaffePCA.jpeg")
plot(jaffe.res.pca, col.ind = jaffe$y, label="none")
dev.off()

mfea.res.pca = PCA(mfeaData[,-241], scale.unit = TRUE, ncp = 2)
jpeg("mfeaPCA.jpeg")
plot(mfea.res.pca, col.ind = mfea$y, label="none")
dev.off()

mnist.res.pca = PCA(mnistData[,-785], scale.unit = TRUE, ncp = 2)
jpeg("mnistPCA.jpeg")
plot(mnist.res.pca, col.ind = mnist$y, label="none")
dev.off()

optidigits.res.pca = PCA(optidigitsData[,-65], scale.unit = TRUE, ncp = 2)
jpeg("optidigitsPCA.jpeg")
plot(optidigits.res.pca, col.ind = optidigits$y, label="none")
dev.off()

usps.res.pca = PCA(uspsData[,-257], scale.unit = TRUE, ncp = 2)
jpeg("uspsPCA.jpeg")
plot(usps.res.pca, col.ind = usps$y, label="none")
dev.off()

################################################################################################################
# Using Nclust we implement different Clustering Algorithms with different methods ( FOR ALL DATASETS ): Kmeans,ward,average,single
################################################################################################################

res.nbclust.jaffe.ward = NbClust(data = jaffeData[,-677], min.nc = 2, max.nc = 20, method = "ward.D", index = "kl")
res.nbclust.jaffe.ward
table(res.nbclust.jaffe.ward$Best.partition, jaffe$y)

res.nbclust.jaffe.kmeans = NbClust(data = jaffeData[,-677], min.nc = 2, max.nc = 20, method = "kmeans", index = "kl")
res.nbclust.jaffe.kmeans
table(res.nbclust.jaffe.kmeans$Best.partition, jaffe$y)

res.nbclust.jaffe.average = NbClust(data = jaffeData[,-677], min.nc = 2, max.nc = 20, method = "average", index = "kl")
res.nbclust.jaffe.average
table(res.nbclust.jaffe.average$Best.partition, jaffe$y)

res.nbclust.jaffe.single = NbClust(data = jaffeData[,-677], min.nc = 2, max.nc = 20, method = "single", index = "kl")
res.nbclust.jaffe.single
table(res.nbclust.jaffe.single$Best.partition, jaffe$y)

res.nbclust.jaffe.complete = NbClust(data = jaffeData[,-677], min.nc = 2, max.nc = 20, method = "complete", index = "kl")
res.nbclust.jaffe.complete
table(res.nbclust.jaffe.complete$Best.partition, jaffe$y)

#######################################################################

res.nbclust.mfea.ward = NbClust(mfeaData[,-241], distance = "euclidean", min.nc=2, max.nc=20, method = "ward.D", index = "kl")
res.nbclust.mfea.ward
# 4 clusters

res.nbclust.mfea.kmeans = NbClust(mfeaData[,-241], distance = "euclidean", min.nc=2, max.nc= 20, method = "kmeans", index = "kl")
res.nbclust.mfea.kmeans
# clusters

res.nbclust.mfea.average = NbClust(mfeaData[,-241], distance = "euclidean", min.nc=2, max.nc= 20, method = "average", index = "kl")
res.nbclust.mfea.average

res.nbclust.mfea.single = NbClust(mfeaData[,-241], distance = "euclidean", min.nc=2, max.nc= 20, method = "single", index = "kl")
res.nbclust.mfea.single

res.nbclust.mfea.complete = NbClust(mfeaData[,-241], distance = "euclidean", min.nc=2, max.nc= 20, method = "complete", index = "kl")
res.nbclust.mfea.complete$Best.nc

###################################################################

res.nbclust.mnist.ward = NbClust(mnistData[,-785], distance = "euclidean", min.nc=2, max.nc= 20, method = "ward.D", index = "silhouette")
res.nbclust.mnist.ward$Best.nc

res.nbclust.mnist.kmeans = NbClust(mnistData[,-785], distance = "euclidean", min.nc=2, max.nc= 20, method = "kmeans", index = "silhouette")
res.nbclust.mnist.kmeans

res.nbclust.mnist.single = NbClust(mnistData[,-785], distance = "euclidean", min.nc=2, max.nc= 20, method = "single", index = "silhouette")
res.nbclust.mnist.single

res.nbclust.mnist.average = NbClust(mnistData[,-785], distance = "euclidean", min.nc=2, max.nc= 20, method = "average", index = "silhouette")
res.nbclust.mnist.average

res.nbclust.mnist.complete = NbClust(mnistData[,-785], distance = "euclidean", min.nc=2, max.nc= 20, method = "complete", index = "silhouette")
res.nbclust.mnist.complete

###################################################################

res.nbclust.optidigits.wardres.nbclust.optidigits.ward = NbClust(optidigitsData[,-65], distance = "euclidean", min.nc=2, max.nc= 20, method = "ward.D", index = "silhouette")
res.nbclust.optidigits.ward$Best.nc

res.nbclust.optidigits.kmeans = NbClust(optidigitsData[,-65], distance = "euclidean", min.nc=2, max.nc= 20, method = "kmeans", index = "silhouette")
res.nbclust.optidigits.kmeans$Best.nc

res.nbclust.optidigits.single = NbClust(optidigitsData[,-65], distance = "euclidean", min.nc=2, max.nc= 20, method = "single", index = "silhouette")

res.nbclust.optidigits.average = NbClust(optidigitsData[,-65], distance = "euclidean", min.nc=2, max.nc= 20, method = "average", index = "silhouette")

res.nbclust.optidigits.complete = NbClust(optidigitsData[,-65], distance = "euclidean", min.nc=2, max.nc= 20, method = "complete", index = "silhouette")

###################################################################

res.nbclust.usps.ward = NbClust(uspsData[,-257], distance = "euclidean", min.nc=2, max.nc= 20, method = "ward.D", index = "silhouette")

res.nbclust.usps.kmeans = NbClust(uspsData[,-257], distance = "euclidean", min.nc=2, max.nc= 20, method = "kmeans", index = "silhouette")

res.nbclust.usps.single = NbClust(uspsData[,-257], distance = "euclidean", min.nc=2, max.nc= 20, method = "single", index = "silhouette")

res.nbclust.usps.average = NbClust(uspsData[,-257], distance = "euclidean", min.nc=2, max.nc= 20, method = "average", index = "silhouette")

res.nbclust.usps.complete = NbClust(uspsData[,-257], distance = "euclidean", min.nc=2, max.nc= 20, method = "complete", index = "silhouette")

################################
# Clustering form ACP Components
#################################

res.hcpc.jaffe = HCPC(jaffe.res.pca, min = 2, max = 20, kk = Inf)
res.hcpc.jaffe
table(res.hcpc.jaffe$data.clust[,677], jaffe$y)

res.hcpc.mfea = HCPC(mfea.res.pca,  min = 2, max = 20, kk = Inf)
res.hcpc.mfea

res.hcpc.mnist = HCPC(mnist.res.pca, min = 2, max = 20, kk = Inf)
res.hcpc.mnist

res.hcpc.optidigits = HCPC(optidigits.res.pca, min = 2, max = 20, kk = Inf)
res.hcpc.optidigits

res.usps.mnist = HCPC(usps.res.pca, min = 2, max = 20, kk = Inf)
res.usps.mnist

################################################
# Confusion Matrix 
################################################

a = table(res.hcpc.jaffe$data.clust$clust, jaffe$y)
summary(a)
confusionMatrix(a)
table(res.nbclust.mfea.ward$Best.partition, mfea$y)

################################################
#  MCLUST 
################################################


nbClusters = 1:20

### jaffe ###

BICS = c()
for (cl in nbClusters) {
  res.mclust.jaffe = Mclust(jaffeData[,-677], G=cl)
  BICS = append(BICS, res.mclust.jaffe$bic)
}
plot(nbClusters, BICS)
res.mclust.jaffe = Mclust(jaffeData[,-677], G= nbClusters)
res.mclust.jaffe$BIC

#### mfea ###

BICS = c()
for (cl in nbClusters) {
  res.mclust.mfea = Mclust(mfeaData[,-241], G=cl)
  BICS = append(BICS, res.mclust.mfea$bic)
}
plot(nbClusters, BICS)
res.mclust.mfea = Mclust(mfeaData[,-241], G= nbClusters)
res.mclust.mfea$BIC

#### mnist ###

BICS = c()
for (cl in nbClusters) {
  res.mclust.mnist = Mclust(mnistData[,-785], G=cl)
  BICS = append(BICS, res.mclust.mnist$bic)
}
res.mclust.mnist = Mclust(mnistData[,-785], G= nbClusters)
res.mclust.mnist$BIC

#### optidigits ###

BICS = c()
for (cl in nbClusters) {
  res.mclust.optidigits = Mclust(optidigitsData[,-65], G=cl)
  BICS = append(BICS, res.mclust.optidigits$bic)
}
res.mclust.optidigits = Mclust(optidigitsData[,-65], G= nbClusters)
res.mclust.optidigits$BIC

#### usps ###

BICS = c()
for (cl in nbClusters) {
  res.mclust.usps = Mclust(uspsData[,-257], G=cl)
  BICS = append(BICS, res.mclust.usps$bic)
}
res.mclust.usps = Mclust(uspsData[,-257], G= nbClusters)
res.mclust.usps$BIC

################################################
# MCLUST DR
################################################

### jaffe ###

res.mclustdr.jaffe = MclustDR(res.mclust.jaffe, lambda = 1)
res.mclustdr.jaffe$class
summary(res.mclustdr.jaffe)
plot(res.mclustdr.jaffe, what = "scatterplot", symbols = 1:2)
plot(res.mclustdr.jaffe, what = "evalues")

### mfea ###

res.mclustdr.mfea = MclustDR(res.mclust.mfea, lambda = 1)
summary(res.mclustdr.mfea)
plot(res.mclustdr.mfea, what = "scatterplot", symbols = 1:2)
plot(res.mclustdr.mfea, what = "evalues")

### mnist ###

res.mclustdr.mnist = MclustDR(res.mclust.mnist, lambda = 1)
summary(res.mclustdr.mnist)
plot(res.mclustdr.mnist, what = "scatterplot", symbols = 1:2)
plot(res.mclustdr.mnist, what = "evalues")

### optidigits ###

res.mclustdr.optidigits = MclustDR(res.mclust.optidigits, lambda = 1)
res.mclustdr.optidigits$x
summary(res.mclustdr.optidigits)
plot(res.mclustdr.optidigits, what = "scatterplot", symbols = 1:2)
plot(res.mclustdr.optidigits, what = "evalues")

### usps ###

res.mclustdr.usps = MclustDR(res.mclust.usps, lambda = 1)
summary(res.mclustdr.optidigits)
plot(res.mclustdr.usps, what = "scatterplot", symbols = 1:2)
plot(res.mclustdr.usps, what = "evalues")

################################################
# NMI ARI INDEX
################################################

### jaffe ###
ari.jaffe.nbclust.ward = adjustedRandIndex(res.nbclust.jaffe.ward$Best.partition, jaffe$y)
ari.jaffe.nbclust.ward

ari.mfea.nbclust.kmeans = adjustedRandIndex(res.nbclust.jaffe.kmeans$Best.partition ,jaffe$y)
ari.mfea.nbclust.kmeans

ari.mfea.nbclust.single = adjustedRandIndex(res.nbclust.jaffe.single$Best.partition, jaffe$y)
ari.mfea.nbclust.single

ari.mfea.nbclust.average = adjustedRandIndex(res.nbclust.jaffe.average$Best.partition, jaffe$y)
ari.mfea.nbclust.average

ari.mfea.nbclust.complete = adjustedRandIndex(res.nbclust.jaffe.complete$Best.partition, jaffe$y)
ari.mfea.nbclust.complete

ari.jaffe.mclust = adjustedRandIndex(res.mclust.jaffe$classification, jaffe$y)
ari.jaffe.mclust

ari.jaffe.mclustdr = adjustedRandIndex(res.mclustdr.jaffe$class, jaffe$y)
ari.jaffe.mclustdr


#### NMI INDEX 

ari.jaffe.mclust = NMI(as.vector(res.mclust.jaffe$classification), as.vector(jaffe$y), variant = c("max")) 
ari.jaffe.mclust

ari.jaffe.mclustdr = NMI(as.vector(res.mclustdr.jaffe$class), as.vector(jaffe$y), variant = c("max"))
ari.jaffe.mclustdr

### mfea ###

ari.mfea.nbclust.ward = adjustedRandIndex(res.nbclust.mfea.ward$Best.partition, mfea$y)

ari.mfea.mclust = adjustedRandIndex(res.mclust.mfea$classification, mfea$y)

ari.mfea.mclustdr = adjustedRandIndex(res.mclustdr.mfea, mfea$y)

### mnist ###
ari.mnist.nbclust = adjustedRandIndex(res.nbclust.mnist.ward, jaffemnisty)

ari.mnist.mclust = adjustedRandIndex(res.mclust.mnist$classification, mnist$y)

ari.mnist.mclustdr = adjustedRandIndex(res.mclustdr.mnist, mnist$y)

### optidigits ###
ari.optidigits.nbclust = adjustedRandIndex(res.nbclust.optidigits.ward, optidigits$y)

ari.optidigits.mclust = adjustedRandIndex(res.mclust.optidigits$classification, optidigits$y)

ari.optidigits.mclustdr = adjustedRandIndex(res.mclustdr.optidigits, optidigits$y)

### usps ###
ari.usps.nbclust = adjustedRandIndex(res.nbclust.jaffe.ward, usps$y)

ari.usps.mclust = adjustedRandIndex(res.mclust.jaffe$classification, usps$y)

ari.usps.mclustdr = adjustedRandIndex(res.mclustdr.jaffe, usps$y)


## TSNE 



## jaffe
tsne <- Rtsne(jaffeData[-677], dims = 2, perplexity=30, verbose=TRUE, max_iter = 500)
plot(tsne$Y)


## mfea ## here we had a problem of duplicates, we chose to ignore it, by Remove duplicates parameter
tsne <- Rtsne(mfeaData[,-241], dims = 2, perplexity=30, verbose=TRUE, max_iter = 500,check_duplicates = FALSE)
plot(tsne$Y)

## mnist
tsne <- Rtsne(mnistData[-785], dims = 2, perplexity=30, verbose=TRUE, max_iter = 500)
plot(tsne$Y)

## optidigits
tsne <- Rtsne(optidigitsData[-677], dims = 2, perplexity=30, verbose=TRUE, max_iter = 500)
plot(tsne$Y)

## usps
tsne <- Rtsne(uspsData[-677], dims = 2, perplexity=30, verbose=TRUE, max_iter = 500)
plot(tsne$Y)

###################################################
##### Deep learning for dimension reduction
##### Calculating NMI and Accuracy and ARI for each Dataset 
####################################################

# Jaffe Dataset 
data_x_norm <- jaffe$X/max(jaffe$X)
# Model
dim= ncol(data_x_norm)
input_d <- layer_input(shape = dim)
#"encoded" is the encoded representation of the input
encoded<- layer_dense(input_d,100  ,activation = "relu")
#encoded<- layer_dense(encoded,500  ,activation = "relu")
encoded<- layer_dense(encoded,50  , activation = "relu")
encoded<- layer_dense(encoded,10 , activation = "relu")
decoded<- layer_dense(encoded,50 ,activation = "relu")
#decoded<- layer_dense(decoded,500  ,activation = "relu")
decoded<- layer_dense(decoded,100 ,activation = "relu")
decoded<- layer_dense(decoded, dim )

autoencoder <- keras_model(input_d, decoded)
encoder <- keras_model(input_d, encoded)

autoencoder  %>% compile(optimizer = "adam", loss = 'mse')

hist= autoencoder %>% fit (data_x_norm, 
                           data_x_norm, 
                           epochs=250,
                           batch_size=32)#,

plot(hist)

encoded_data_x <- predict(encoder, data_x_norm)

res.mclust_encoded = Mclust(encoded_data_x)

res.mclust_encoded$bic

# res.mclust_encoded$bic
# [1] -3007,412

table(res.mclust_encoded$classification, jaffe$y)



#### MFEA  ######

data_x_norm <- mfea$X/max(mfea$X)

# Model
dim= ncol(data_x_norm)
input_d <- layer_input(shape = dim)

encoded<- layer_dense(input_d,150  ,activation = "relu")
#encoded<- layer_dense(encoded,500  ,activation = "relu")
encoded<- layer_dense(encoded,70  , activation = "relu")
encoded<- layer_dense(encoded,10 , activation = "relu")
decoded<- layer_dense(encoded,70 ,activation = "relu")
#decoded<- layer_dense(decoded,500  ,activation = "relu")
decoded<- layer_dense(decoded,150 ,activation = "relu")
decoded<- layer_dense(decoded, dim )

autoencoder <- keras_model(input_d, decoded)
encoder <- keras_model(input_d, encoded)

autoencoder  %>% compile(optimizer = "adam", loss = 'mse')

hist= autoencoder %>% fit (data_x_norm, 
                           data_x_norm, 
                           epochs=250,
                           batch_size=32)#,

plot(hist)

encoded_data_x <- predict(encoder, data_x_norm)

res.mclust_encoded = Mclust(encoded_data_x)

res.mclust_encoded
res.mclust_encoded$bic

a = table(res.mclust_encoded$classification, mfea$y)

a
NMI(as.vector(res.mclust_encoded$classification),as.vector(mfea$y))

adjustedRandIndex(res.mclust_encoded$classification,mfea$y)
#### Mnist ######

data_x_norm <- mnist$X/max(mnist$X)

# Model
dim= ncol(data_x_norm)
input_d <- layer_input(shape = dim)

encoded<- layer_dense(input_d,150  ,activation = "relu")
#encoded<- layer_dense(encoded,500  ,activation = "relu")
encoded<- layer_dense(encoded,70  , activation = "relu")
encoded<- layer_dense(encoded,10 , activation = "relu")
decoded<- layer_dense(encoded,70 ,activation = "relu")
decoded<- layer_dense(decoded,150 ,activation = "relu")
decoded<- layer_dense(decoded, dim )

autoencoder <- keras_model(input_d, decoded)
encoder <- keras_model(input_d, encoded)

autoencoder  %>% compile(optimizer = "adam", loss = 'mse')

hist= autoencoder %>% fit (data_x_norm, 
                           data_x_norm, 
                           epochs=250,
                           batch_size=32)#,

plot(hist)

encoded_data_x <- predict(encoder, data_x_norm)

res.mclust_encoded = Mclust(encoded_data_x)

res.mclust_encoded
res.mclust_encoded$bic

a = table(res.mclust_encoded$classification, mnist$y)

a
NMI(as.vector(res.mclust_encoded$classification),as.vector(mnist$y))

adjustedRandIndex(res.mclust_encoded$classification,mnist$y)


###### Optidigits #######


data_x_norm <- optidigits$X/max(optidigits$X)

# Model
dim= ncol(data_x_norm)
input_d <- layer_input(shape = dim)
#"encoded" is the encoded representation of the input
encoded<- layer_dense(input_d,40  ,activation = "relu")
#encoded<- layer_dense(encoded,500  ,activation = "relu")
encoded<- layer_dense(encoded,20  , activation = "relu")
encoded<- layer_dense(encoded,10 , activation = "relu")
decoded<- layer_dense(encoded,20 ,activation = "relu")
#decoded<- layer_dense(decoded,500  ,activation = "relu")
decoded<- layer_dense(decoded,40 ,activation = "relu")
decoded<- layer_dense(decoded, dim )

autoencoder <- keras_model(input_d, decoded)
encoder <- keras_model(input_d, encoded)

autoencoder  %>% compile(optimizer = "adam", loss = 'mse')

hist= autoencoder %>% fit (data_x_norm, 
                           data_x_norm, 
                           epochs=250,
                           batch_size=32)#,

plot(hist)

encoded_data_x <- predict(encoder, data_x_norm)

res.mclust_encoded = Mclust(encoded_data_x)

res.mclust_encoded
res.mclust_encoded$bic

a = table(res.mclust_encoded$classification, optidigits$y)

a
NMI(as.vector(res.mclust_encoded$classification),as.vector(optidigits$y))

adjustedRandIndex(res.mclust_encoded$classification,optidigits$y)


#### USPS ######

data_x_norm <- usps$X/max(usps$X)

# Model
dim= ncol(data_x_norm)
input_d <- layer_input(shape = dim)

encoded<- layer_dense(input_d,150  ,activation = "relu")
#encoded<- layer_dense(encoded,500  ,activation = "relu")
encoded<- layer_dense(encoded,70  , activation = "relu")
encoded<- layer_dense(encoded,10 , activation = "relu")
decoded<- layer_dense(encoded,70 ,activation = "relu")
#decoded<- layer_dense(decoded,500  ,activation = "relu")
decoded<- layer_dense(decoded,150 ,activation = "relu")
decoded<- layer_dense(decoded, dim )

autoencoder <- keras_model(input_d, decoded)
encoder <- keras_model(input_d, encoded)

autoencoder  %>% compile(optimizer = "adam", loss = 'mse')

hist= autoencoder %>% fit (data_x_norm, 
                           data_x_norm, 
                           epochs=250,
                           batch_size=32)#,

plot(hist)

encoded_data_x <- predict(encoder, data_x_norm)

res.mclust_encoded = Mclust(encoded_data_x)

res.mclust_encoded
res.mclust_encoded$bic

a = table(res.mclust_encoded$classification, usps$y)

a

NMI(as.vector(res.mclust_encoded$classification),as.vector(usps$y))

adjustedRandIndex(res.mclust_encoded$classification,usps$y)


####### END OF CODE ########
