# Exploring convolutional neural networks using MNIST handwriting data

# Clear environment
rm(list=ls())

# Set working directory and load libraries
setwd("~")
library(keras)

# Load data. This has both training and test data. Here x is grayscale image
# data with shape of 28 x 28 and y is a label denoting the digit (0-9) described
# by the image.
mnist <- keras::dataset_mnist()

# Specify the model. We'll try a flat model first, not a CNN.
model <- keras::keras_model_sequential() %>%
  keras::layer_flatten(input_shape = c(28, 28)) %>%
  keras::layer_dense(
    units = 32,
    activation = "relu") %>%
  keras::layer_dense(units = 5, activation = "relu") %>%
  keras::layer_dense(units = 10, activation = "softmax")

# Compile the model.
model %>% keras::compile(
  loss = "sparse_categorical_crossentropy",
  optimizer = "adam",
  metrics = c("accuracy"))

# Train model across 50 epochs
model %>% keras::fit(
  mnist$train$x / 255,
  mnist$train$y,
  epoch = 50)

# Evaluate the model
model %>% keras::evaluate(mnist$test$x, mnist$test$y)
predictions <- model %>% predict(mnist$test$x)
table(apply(predictions, 1, which.max) - 1, mnist$test$y)

# Looks great, but let's try a CNN
model.cnn <- keras::keras_model_sequential() %>%
  keras::layer_conv_2d(
    filters = 20,
    kernel_size = c(3, 3),
    activation = 'relu',
    input_shape = c(28, 28, 1)) %>%
  keras::layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  keras::layer_flatten() %>%
  keras::layer_dense(
    units = 32,
    activation = "relu") %>%
  keras::layer_dense(units = 10, activation = 'softmax') %>%
  keras::compile(
    loss = "sparse_categorical_crossentropy",
    optimizer = "adam",
    metrics = c("accuracy"))

# Fit the model
array.exp <- array(mnist$train$x / 255, dim = c(dim(mnist$train$x), 1))
model.cnn %>% fit(mnist$test$x, mnist$test$y, epochs = 10)

# Evaluate the model
model.cnn %>% keras::evaluate(mnist$test$x, mnist$test$y)
predictions.cnn <- model.cnn %>% predict(mnist$test$x)
table(apply(predictions.cnn, 1, which.max) - 1, mnist$test$y)

# Cool, flat neural network worked great, but the CNN worked even better.