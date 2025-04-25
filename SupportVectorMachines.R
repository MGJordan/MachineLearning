# Exploring Support Vector Machines using the Heart dataset from
# An Introduction to Statistical Learning
# https://www.statlearning.com/resources-second-edition

# Clear environment
rm(list=ls())

# Set working directory and load libraries
setwd("~")
library(ggplot2)
library(e1071)
library (pROC)
library(dplyr)

# Import Heart data
data <- read.csv("./Downloads/Heart.csv")

# SVMs are a classification tool that is an extension of the Maximal Margin
# Classifier and the Support Vector Classifier. Consider a two-dimensional
# dataset in which data are divided into a class with one level. Imagine that
# the two classes can be perfectly separated by a line with all data points from
# the classes falling on the opposite side of the line. The distance between
# the data points and the line is called the margin. The Maximal Margin
# Classifer is the line for which the margin is highest. The idea is that this
# line should outperform other lines with smaller margins because a large margin
# allows for more generalizeablility. We can take this example into more
# dimensions by considering a hyperplane rather than a line. Of course, this
# technique assumes that a separating hyperplane exists for a given dataset. It
# might not! In which case, an MCM won't work. Luckily we can extend MCMs into
# Support Vector Classifiers. An interesting property of MCMs is that the
# maximal hyperplane only depends on the data points located on the margins,
# which we call support vectors (since they support the maximal margin
# hyperplane in the sense that if they move then the hyperplane moves). A SVC
# allows these support vectors to violate the margin and even the maximal
# hyperplane. Each support vector is allowed an error term such that if e > 0,
# the vector violates the margin and if e > 1 it violates the hyperplane. The
# degree of total violation is tuned by the factor C.

# Fantastic. So, what are Support Vector Machines? They are an extension of SVCs
# in which the maximal hyperplane does not need to be linear, but can be 
# polynomial, radial, etc. They do this using kernels. So, let's try this.

# Let's first test a linear application of this on some two-dimensional data.
# We'll classify whether the car is automatic or manual (am) using the rear
# axle ratio (drat) and weight (wt).
data(mtcars)
cars.df <- as.data.frame(mtcars[, c("drat", "wt", "am")])
cars.df$am <- as.factor(cars.df$am)

# Some overlap, but they should be separable based on a linear SVC.
ggplot2::ggplot(cars.df, aes(x = drat, y = wt, color = am)) +
  ggplot2::geom_point()

# Fit an SVM with a linear kernal and an arbitrary cost value. 
svm.lm <- e1071::svm(
  am ~ .,
  data = cars.df,
  kernel = "linear",
  cost = 10,
  scale = TRUE)

# Looks like we have 11 support vectors. 
summary(svm.lm)
plot(svm.lm, cars.df)

# Let's use cross validation find out the cost value that gives us the lowest
# error rate.
tune.svm.lm <- e1071::tune(
  svm,
  am ~ .,
  data = cars.df,
  kernel = "linear",
  ranges = list(cost = c(0.001, 0.01, 0.1, 1, 5, 10, 100, 1000)))
tune.svm.lm$best.model # cost = .1

# Evaluate. A lower cost value with wider margins and more support vectors (20)
# is optimal.
svm.lm.opt <- e1071::svm(
  am ~ .,
  data = cars.df,
  kernel = "linear",
  cost = .1,
  scale = TRUE)
summary(svm.lm.opt)
plot(svm.lm.opt, cars.df)

# Okay, let's try this out with higher dimensional data. We could recode the
# factors as binary variables, but we'll drop for ease. Recode AHD (whether
# patient has heart disease), drop NAs.
heart.df <- as.data.frame(read.csv("./Downloads/Heart.csv"))
heart.df <- heart.df[, !names(heart.df) %in% c("X", "ChestPain", "Thal")]
heart.df$AHD <- as.factor(ifelse(heart.df$AHD == "No", -1, 1))
heart.df <- na.omit(heart.df)

# Split the data into training and testing sets.
train <- dplyr::sample_frac(heart.df, .1)
test <- dplyr:: setdiff(heart.df, train)

# Fit an SVM with a radial kernel and arbitrarily picked cost and gamma. Here
# gamma determines how far the influence of a single data point reaches. When
# it's high, each data point influences a small region and when it's low each
# data point influences a large region. High values can lead to overfitting and
# low to underfitting. A low gamma results in a smooth decision surface.
svm.r <- e1071::svm(
  AHD ~ .,
  data = train,
  kernel = "radial",
  cost = 1,
  gamma = 1,
  scale = TRUE)
summary(svm.r)

# We can plot this in two dimensions, but it's not that useful.
plot(svm.r, train, RestBP ~ MaxHR)

# Anyway, let's see how this performs on the test data. Wow, so bad! 147 out of
# 269 data points are incorrectly classified.
table(predict = predict(svm.r, test), truth = test$AHD)

# Not a big surprise I suppose since we picked our parameters arbitrarily. Let's
# use cross-validation to pick optimal parameters to minimize errors.
# Grid search ranges from:
# https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf
tune.svm.r <- e1071::tune(
  svm,
  AHD ~ .,
  data = train,
  kernel = "radial",
  scale = TRUE,
  ranges = list(
    cost = 2^(-5:15),
    gamma = 2^(-15:3)))
tune.svm.r$best.model$cost # 32768
tune.svm.r$best.model$gamma # 0.0001220703

# Cool, let's see how it performs.
svm.r.opt <- e1071::svm(
  AHD ~ .,
  data = train,
  kernel = "radial",
  cost = tune.svm.r$best.model$cost,
  gamma = tune.svm.r$best.model$gamma,
  scale = TRUE)
table(predict = predict(svm.r.opt, test), truth = test$AHD)

# Way better, only 63 out of 269 incorrectly classified.

# Let's look at the ROC curves for both of the SVMs we fit.
roc_score <- pROC::roc(
  test$AHD,
  as.numeric(predict(svm.r, newdata = test, type = "response")))
roc_score.opt <- pROC::roc(
  test$AHD,
  as.numeric(predict(svm.r.opt, newdata = test, type = "response")))
par(mfrow = c(1, 2))
plot(roc_score, main = "Model w/ Random Parameters")
plot(roc_score.opt, main = "Model w/ Optimal Parameters")

# Yup, no surprise, optimal model looks a lot better, although still not
# amazing. Note that the threshold that is being varied here is the decision
# threshold, which is how close data points are to the decision boundary, which
# is the hyperplane that was fit using the support vectors.
