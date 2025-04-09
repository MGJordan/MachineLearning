# Exploring K-means clustering using the Iris dataset

# Clear environment
rm(list=ls())

# Set working directory and load libraries
setwd("~")
library(splits)
library(dplyr)
library(ggplot2)

# Columns are in different units, so we'll z-standardize them so that no one
# variable has a greater influence. Dropping the species column.
iris.s <- as.data.frame(scale(subset(iris, select = -c(Species))))

# Inspect the data
hist(iris.s$Sepal.Length) # kinda normal
hist(iris.s$Sepal.Width) # really pretty normal 
hist(iris.s$Petal.Length) # not normal, kinda bimodal with left skew
hist(iris.s$Petal.Width) # not normal, kind bimodal with left skew

# Normality shouldn't matter for k-means clustering

# Create a distance matrix using Euclidean distance
iris.d <- dist(iris.s, method = "euclidean")

# Use the DD-weighted gap statistic to pick the number of clusters
gap.stat <- splits::ddwtGap(iris.d)
with(
  gap.stat,
  plot(colMeans(DDwGap),
       pch = 15,
       type = 'b',
       ylim = extendrange(colMeans(DDwGap), f = 0.2),
       xlab = "Number of Clusters", ylab = "Weighted Gap Statistic"))
gap.stat$mnGhatWG # mean number of well separated clusters based on wGap
gap.stat$mnGhatDD # mean number of well separated clusters based on DDwGap

# DDwGap has it at 2, but wGap has it at 2.6. Seems like it obviously should be
# 3 because the original data contains three species. Fairly close.

# Run k-means clustering with 3 clusters and 10 starts
k.means.10 <- kmeans(iris.s, centers = 3, nstart = 10)

# Evaluate k-means clustering success
results.10 <- data.frame(
  Species = iris$Species,
  Cluster = k.means.10$cluster)
results.10 %>%
  dplyr::group_by(Species, Cluster) %>%
  summarise(n = n())

# Not bad, setosa separated well whereas ~ 1/5 of versicolor and virginica are
# incorrect. Let's try with more starts.

# Run k-means clustering with 3 clusters and 100 starts
k.means.100 <- kmeans(iris.s, centers = 3, nstart = 100)

# Evaluate k-means clustering success
results.100 <- data.frame(
  Species = iris$Species,
  Cluster = k.means.100$cluster)
results.100 %>%
  dplyr::group_by(Species, Cluster) %>%
  summarise(n = n())

# No real difference