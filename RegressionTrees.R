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

# Residual mean deviance is 5.7. I.e., total residual deviance (sum of squared
# errors) / number of observation minus number of leaves. Seems kind of high
# to me given that mean MPG is ~ 23.

# Let's check how the model performs outside the training data
cor.test(predict(rt, data.f[-training,]), data.f$mpg[-training])

# Hey, 90% is pretty good!

# Check cross-validation of model
plot(cv.tree(rt))