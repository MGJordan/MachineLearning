# Exploring hierarchical cluster analysis using US voting data

# Clear environment
rm(list=ls())

# Set working directory and load libraries
setwd("~")
library(cluster)
library(splits)
library(ggplot2)

# Create object with the proportion of votes the Republican party received
# by state and year.
votes <- na.omit(cluster::votes.repub)

# Because percentages aren't normally distributed, we'll use a logit
# transformation
logit <- function(x) log(x / (1-x))
votes.t <- logit(votes / 100) # convert to %

# Create a distance matrix and cluster using unweighted pair group method with
# arithmetic mean, single linkage, and complete linkage
votes.d <- dist(votes.t, method = "euclidean")
upgma <- hclust(votes.d, method = "average") # based on cluster average
single.link <- hclust(votes.d, method = "single") # based on nearest cluster
complete.link <- hclust(votes.d, method = "complete") # based on furthest c

# Plot clusters
plot(upgma) # space conserving
plot(single.link) # space contracting
plot(complete.link) # space expanding

# Fair enough, but how should we pick how many clusters into which to cut the
# data?
# One method is the DD-weighted gap statistic
# (https://doi.org/10.1111/j.1541-0420.2007.00784.x)
# Compares the average distance among points within a given cluster for a given
# number of clusters with the distances if there were one fewer or one more
# clusters. Basically you're trying to maximize low intra cluster distance
# vs high inter cluster distance.

# Let's test it
gap.stat <- splits::ddwtGap(votes.d)
with(
  gap.stat,
  plot(colMeans(DDwGap),
  pch = 15,
  type = 'b',
  ylim = extendrange(colMeans(DDwGap), f = 0.2),
  xlab = "Number of Clusters", ylab = "Weighted Gap Statistic"))
gap.stat$mnGhatDD

# 5 clusters is optimal, so we can cut the data there
upgma.cut <- cutree(upgma, k = 5)
single.link.cut <- cutree(single.link, k = 5)
complete.link.cut <- cutree(complete.link, k = 5)

# No difference between linkage methods
