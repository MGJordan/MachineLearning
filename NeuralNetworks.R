# Exploring neural networks using boardgame data from BoardGameGeek
# https://www.kaggle.com/datasets/threnjen/board-games-database-from-boardgamegeek/data?select=games.csv

# Clear environment
rm(list=ls())

# Set working directory and load libraries
setwd("~")
library(keras)

# Import data
bg <- as.data.frame(read.csv("./Downloads/games.csv"))

# Create objects with just columns of interest. We'll predict game rating based
# on some explanatory variables that might reasonably be thought to influence
# it. Omit NAs.
bg.f <- bg[
  ,
  c("AvgRating", "GameWeight", "MinPlayers", "MaxPlayers", "ComAgeRec",
    "LanguageEase", "BestPlayers", "MfgPlaytime", "ComMinPlaytime",
    "ComMaxPlaytime", "MfgAgeRec", "NumAlternates", "IsReimplementation",
    "Kickstarted")]
bg.f <- na.omit(bg.f)

# Divide data into explanatory and response matrices, scale, and derive training
# subset.
exp <- as.matrix(scale(bg.f[, 2:14]))
resp <- as.numeric(scale(bg.f[, 1]))
training <- sample(nrow(exp), nrow(exp) / 10)

# Specify the model. We have 14 explanatory variables, so let's go with 15
# nodes in the first layer and 8 nodes in the second layer.
# Use ReLU (rectified linear unit) activation function.
model <- keras::keras_model_sequential() %>%
  keras::layer_dense(units = 50, activation = 'relu', input_shape = 13) %>%
  keras::layer_dense(units = 25, activation = 'relu') %>%
  keras::layer_dense(units = 1)

# Compile the model. Use MSE loss function.
model %>% keras::compile(
  loss = 'mean_squared_error',
  optimizer = keras::optimizer_rmsprop(),
  metrics = c('mean_squared_error'))

# Train model across 200 epochs
model %>% keras::fit(
  exp[training, ],
  resp[training],
  epoch = 200)

# 200 epochs seems reasonable, it starts to plateau

# See how model performs
plot(predict(model, exp[-training,])[,1] ~ resp[-training])
cor.test(predict(model, exp[-training,])[,1], resp[-training])

# Odd, as the true response increases, the prediction is unchanged. Correlation
# is ~.35, decent, not amazing. Let's try another activation function.

# Define a model using softmax activation function and same number of layers and
# nodes.
model2 <- keras::keras_model_sequential() %>%
  keras::layer_dense(units = 50, activation = 'softmax', input_shape = 13) %>%
  keras::layer_dense(units = 25, activation = 'softmax') %>%
  keras::layer_dense(units = 1)

# Compile the model. Use MSE loss function.
model2 %>% keras::compile(
  loss = 'mean_squared_error',
  optimizer = keras::optimizer_rmsprop(),
  metrics = c('mean_squared_error'))

# Train model across 200 epochs
model2 %>% keras::fit(
  exp[training, ],
  resp[training],
  epoch = 200)

# Again 200 epochs seems reasonable.

# See how model performs
plot(predict(model2, exp[-training,])[,1] ~ resp[-training])
cor.test(predict(model2, exp[-training,])[,1], resp[-training])

# Much better correlation, .63. Also relationship between prediction and true
# response is roughly linear.