# Exploring Ridge and LASSO (Least Absolute Shrinkage and Selection Operator)
# regression.
# Data originally from Quinlan (1993) “Combining Instance-Based and Model-Based
# Learning” in Proceedings on the Tenth International Conference of Machine
# Learning, 236–243
# http://archive.ics.uci.edu/dataset/9/auto+mpg

# Clear environment
rm(list=ls())

# Set working directory and load libraries
setwd("~")
library(glmnet)
library(dplyr)

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
data.f$origin <- as.integer(data.f$origin) # not treating as factor

# These are types of regressions that incorporate regularization.
# Regularization reduces overfitting by introducing bias into a model during
# training in order to increase generalizeability. It can also be used to
# understand which variables are most important and address multicollinearity.
# It includes a regularization term tuned by the parameter lambda into the loss
# function as we will see below.

# In a typical regression such as OLS (although these techniques also work for
# maximum likelihood methods), we minimize the sum of the squared residuals
# (i.e., differences between the predicted regression and the data). In ridge
# regression we minimize the sum of the squared residuals plus the sum of the 
# squared explanatory variable coefficients times lambda. The basic idea is that
# we want to penalize model complexity, introducing a trade-off between
# minimizing the sum of squared residuals and minimizing model complexity. We
# pick a value for lamba that results in the lowest value for this overall loss
# function.

# Fit a ridge regression using glmnet. Omit factor variables to keep things
# simple.
rr <- glmnet::glmnet(
  x = as.matrix(data.f[, 2:6]),
  y = as.matrix(data.f[, 1]),
  alpha = 0, # ridge regression
  standardize = TRUE, # need to z-standardize since variable units differ
  lambda = 10^seq(10, -2, length = 100)) # fit a large range of λ values

# Examine ridge regression. We can see that as lambda increases from 0
# explanatory variable coefficients increase. Some variables increase much
# faster than others, indicating they are more correlated with the response
# variable. We can also print out the number of variables with a positive
# coefficient and the % of deviance explained at various lambda values.
plot(rr)
print(rr)

# So, how do we pick the optimal lambda value? We can use k-fold cross
# validation. Divide data into k random subsets, pick one subset as the test
# subset, fit the model on the remaining data, evaluate the model on the test
# subset, perform this across all combinations, average the evaluations.

# There's a built in glmnet function for this.
rr.cv <- glmnet::cv.glmnet(
  x = as.matrix(data.f[, 2:6]),
  y = as.matrix(data.f[, 1]),
  alpha = 0,
  nfolds = 10, # 10 folds = 10 subsets
  standardize = TRUE,
  lambda = 10^seq(10, -2, length = 100))
rr.cv$lambda.min

# Refit our model using the optimal lambda
rr.opt <- glmnet::glmnet(
  x = as.matrix(data.f[, 2:6]),
  y = as.matrix(data.f[, 1]),
  alpha = 0,
  standardize = TRUE,
  lambda = rr.cv$lambda.min) # optimal lambda

# Extract coefficients at optimal lambda. Number of cylinders is most
# determinative.
predict(rr.opt, type = "coefficients", s = rr.cv$lambda.min)

# Cool enough, but let's verify that we're actually reducing our error by using
# ridge regression. We'll split our original data into a training and test set.
# Then we'll fit a ridge regression and an OLS on the training data and evaluate
# versus the test data.
train <- dplyr::sample_frac(data.f[, 1:6], .1)
test <- dplyr:: setdiff(data.f[, 1:6], train)
rr2 <- glmnet::glmnet(
  x = as.matrix(train[, 2:6]),
  y = as.matrix(train[, 1]),
  alpha = 0,
  standardize = TRUE)
lr <- lm(mpg ~ ., data = train)
lr.pred <- predict(lr, newdata = test)
rr2.pred.nonoptimal.lambda <- predict(rr2, s = 4, newx = as.matrix(test[, 2:6]))
rr2.pred.optimal.lambda <- predict(
  rr2,
  s = rr.cv$lambda.min,
  newx = as.matrix(test[, 2:6]))

# Examine MSE
mean((lr.pred - as.matrix(test[, 1]))^2) # 20.11541
mean((rr2.pred.nonoptimal.lambda - as.matrix(test[, 1]))^2) # 20.19551
mean((rr2.pred.optimal.lambda - as.matrix(test[, 1]))^2) # 19.31408

# Ridge regression with non-optimal lamba performs worse than linear regression,
# but at the optimal lambda it performs better (although not by much in this
# case).

# Cool enough, let's try LASSO regression! This is similar to ridge regression
# except rather than including lambda times the sum of the squared explanatory
# variable coefficients in the loss function, it includes the modulus of the
# explanatory variables coefficients in the loss function. They're similar,
# but in ridge regression coefficients won't be reduced to zero, whereas they
# can be in LASSO regression, so it's useful for feature selection. 

# Fit a lasso regression
lassor <- glmnet::glmnet(
  x = as.matrix(data.f[, 2:6]),
  y = as.matrix(data.f[, 1]),
  alpha = 1, # lasso regression
  standardize = TRUE,
  lambda = 10^seq(10, -2, length = 100))
plot(lassor) # Some variables are 0 until lambda is great enough

# Get optimal lambda using cross validation
lassor.cv <- glmnet::cv.glmnet(
  x = as.matrix(data.f[, 2:6]),
  y = as.matrix(data.f[, 1]),
  alpha = 1,
  nfolds = 10,
  standardize = TRUE,
  lambda = 10^seq(10, -2, length = 100))
lassor.cv$lambda.min

# Fit it on the training data and evaluate using test data
lassor2 <- glmnet::glmnet(
  x = as.matrix(train[, 2:6]),
  y = as.matrix(train[, 1]),
  alpha = 1,
  standardize = TRUE)
lassor2.pred.nonoptimal.lambda <- predict(
  lassor2,
  s = 4,
  newx = as.matrix(test[, 2:6]))
lassor2.pred.optimal.lambda <- predict(
  lassor2,
  s = lassor.cv$lambda.min,
  newx = as.matrix(test[, 2:6]))

# LASSO with optimal lambda performs very similarly to ridge regression
mean((lassor2.pred.nonoptimal.lambda - as.matrix(test[, 1]))^2) # 33.07709
mean((lassor2.pred.optimal.lambda - as.matrix(test[, 1]))^2) # 19.37541

# It also has the advantage of simplifying our model. Acceleration is now
# zeroed out.
predict(lassor, type = "coefficients", s = lassor.cv$lambda.min)

# We can also do Elastic Net regression, which simply combines Ridge and LASSO
# regression, incorporating both the sum of the squared explanatory variable
# coefficients and the sum of the modulus of the explanatory variable
# coefficients into the loss function and uses the parameter alpha to weight
# each. We won't explore that here, but it should perform at least as well as
# Ridge or LASSO regression since these are really special cases of Elastic Net
# regression.