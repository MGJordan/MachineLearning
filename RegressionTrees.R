# Exploring regression trees using auto mpg data
# Data originally from Quinlan (1993) “Combining Instance-Based and Model-Based
# Learning” in Proceedings on the Tenth International Conference of Machine
# Learning, 236–243
# http://archive.ics.uci.edu/dataset/9/auto+mpg

# Clear environment
rm(list=ls())

# Set working directory and load libraries
setwd("~")
library(tree)
library(randomForest)
library(gbm)

# Import auto mpg data
data <- read.table(
  "./Downloads/auto+mpg/auto-mpg.data",
  sep = "",
  header = FALSE)
names(data) <- c(
  "mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration",
  "modelyear", "origin", "carname")

# Filter data to exclude irrelevant columns and missing data
data.f <- data[!data$horsepower == "?", -which(names(data) %in% c("carname"))]

# Set variable class
data.f$mpg <- as.double(data.f$mpg)
data.f$cylinders <- as.integer(data.f$cylinders)
data.f$displacement <- as.double(data.f$displacement)
data.f$horsepower <- as.double(data.f$horsepower)
data.f$weight <- as.double(data.f$weight)
data.f$acceleration <- as.double(data.f$acceleration)
data.f$modelyear <- as.integer(data.f$modelyear) # treat year as continuous
data.f$origin <- as.factor(data.f$origin)

# Pick a random half of the data as training data and fit a regression tree
training <- sample(nrow(data.f), nrow(data.f)/2)
rt <- tree::tree(mpg ~ ., data = data.f[training, ])

# Here we're predicting the response variable miles per gallon as a regression
# tree of 7 variables (some continuous, some discrete, some categorical)
# The regression tree splits data based on which splitting point for which
# variable minimizes the overall RSS (residual sum of squares) in the model
# (each data point is compared vs the mean of the response variable in its 
# grouping). It's computationally infeasible to consider ever possible
# permutation, so we use a top down greedy (i.e., taking the best split at that
# node) approach called recursive binary splitting.

# Examine the regression tree
plot(rt)
text(rt)

# Model shows that engine displacement is the most determinative variable for
# mpg. Makes sense, this is essentially just engine size and smaller engines
# are more efficient. For smaller engines, the model year is the most
# determinative variable. This is a bit surprising since it has no causal
# impact on engine efficiency, but the split is ~1978, which makes sense as
# (according to Gemini) the most significant increase in engine efficiency
# occurred in the 1970s as a result of oil price shocks and government
# regulation. For larger engines, horsepower is more significant and then
# model year.

# Anyway, is my model any good? Let's look at the stats.
rt
summary(rt)

# Residual mean deviance is 9.1. I.e., total residual deviance (sum of squared
# errors) / number of observation minus number of leaves. Seems kind of high
# to me given that mean MPG is ~ 23.

# Let's check how the model performs outside the training data
cor.test(predict(rt, data.f[-training,]), data.f$mpg[-training])

# Hey, 88% correlation is pretty good!

# Evaluate residual deviance at different numbers of tree nodes
plot(cv.tree(rt))

# Looks like we could prune this tree without a huge increase in deviance.

# This actually all looks pretty good, but another technique to consider is
# bagged regression trees. The model above doesn't seem overfit, but bagged
# regression trees deal with overfitting by dividing training data into random
# subsets, fitting regression trees to these replicates of training data, and
# averaging across models. The basic idea here is that by averaging across
# models you reduce the variance among models and so reduce overfitting. I think
# the intuitive explanation here is that there is a certain probability of bad
# overfitting that comes with working with one subset of the data and averaging
# across various subsets reduces the extremes of that distribution.

# Build a bagged regression tree
bart <- randomForest::randomForest(
  mpg ~ .,
  data = data.f[training, ],
  mtry = ncol(data.f) -1) # Determines how many variables to consider in each
                        # split. Here we include all variables, so making this a
                        # bagged regression tree instead of a random forest.

# Let's see if our correlation went up
cor.test(predict(bart, data.f[-training,]), data.f$mpg[-training])

# It did! 92%. Not a gigantic difference in this case, but notable.

# Let's also try out a random forest model where we vary how many variables are
# used in each replicate. This reduces the correlation between the bootstrapped
# replicates, which should reduce overfitting even more and give us a higher
# correlation in the final model. The basic idea is that the less similar each
# replicate is to the other replicates, the more variance among replicates you
# reduce when you average.
rf <- randomForest::randomForest(
  mpg ~ .,
  data = data.f[training, ],
  importance = TRUE)

# Note that importance is the average decrease in the mean squared error each
# time a split is fit to a given variable. Sort of the explanatory power of
# each variable. Looks like modelyear is most important.
randomForest::importance(rf)

# Anyway, did we increase correlation?
cor.test(predict(rf, data.f[-training,]), data.f$mpg[-training])

# Interesting, it actually went down extremely slightly (< 1%) vs the bagged
# tree.

# Okay, last type of regression tree we'll try out, boosted regression trees.
# They're similar to bagged and random forest models in that they fit multiple
# models, but in this case each model is iteratively fit on the residuals of the
# previous model. This is cool because it incorporates models fit to the
# variation not accounted for by previous models, but results are a little hard
# to interpret. 
# We'll use a gaussian distribution because the data is continuous and normal-ish.
# We can control the relative weight in final results of earlier vs later models
# through the shrinkage parameter (lower values give later models more weight). 
bort <- gbm::gbm(mpg ~ ., data = data.f[training,] , distribution = "gaussian")
bort2 <- gbm::gbm(
  mpg ~ .,
  data = data.f[training,],
  distribution = "gaussian",
  shrinkage = .01)
bort3 <- gbm::gbm(
  mpg ~ .,
  data = data.f[training,],
  distribution = "gaussian",
  shrinkage = 1)

# Did we increase correlation?
cor.test(predict(bort, data.f[-training,]), data.f$mpg[-training])
cor.test(predict(bort2, data.f[-training,]), data.f$mpg[-training])
cor.test(predict(bort3, data.f[-training,]), data.f$mpg[-training])

# Not really. The default shrinkage parameter of .1 produces the best result,
# extremely close to the bagged regression tree, but slightly under it.

# Checking out the variables, horsepower or displacement is the most significant
# under the boosted regression trees. 
summary(bort)
summary(bort2)
summary(bort3)
