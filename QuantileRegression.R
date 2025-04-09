# Exploring quantile regression and its advantages over linear regression
# Based on: https://library.virginia.edu/data/articles/getting-started-with-quantile-regression

# Clear environment
rm(list=ls())

# Set working directory and load libraries
setwd("~")
library(quantreg)
library(ggplot2)

# Create dataset that violates OLS assumption of constant variance
x <- seq(0, 100, length.out = 100)        # independent variable
sig <- 0.1 + 0.05 * x                     # non-constant variance
b_0 <- 6                                  # true intercept
b_1 <- 0.1                                # true slope
set.seed(1)                               # make the next line reproducible
e <- rnorm(100, mean = 0, sd = sig)       # normal random error with non-constant variance
y <- b_0 + b_1 * x + e                    # dependent variable
data <- data.frame(x, y)

# Plot to visualize, confirm heteroskedasticity
ggplot2::ggplot(data, aes(x, y)) +
  geom_point()

# Add linear regression with confidence interval to plot
ggplot2::ggplot(data, aes(x, y)) +
  ggplot2::geom_point() +
  ggplot2::geom_smooth(method="lm")

# Create linear regression model and inspect diagnostics
lr <- lm(y ~ x, data = data)
par(mfrow = c(2, 2))
plot(lr)

# Residuals vs Fitted: Relationship looks linear
# Q-Q Residuals: Residuals fairly normally distributed except for extremes of
# the distribution
# Scale-Location: Heteroskedasticity evident
# Residuals vs Leverage: Some values with a high cook's distance, but no obvious
# values to throw away.

# Just going through the motions here, this is all expected because, you know,
# we created the data.

# So, in our case, linear regression is an unbiased estimator, but statistical
# tests are unreliable as the standard error is unreliable.

# Let's test a quantile regression for the 80% percentile
qr8 <- quantreg::rq(y ~ x, data = data, tau = 0.8)
summary(qr8)

# Interesting, confidence intervals estimated using non-parametric rank test
# http://www.econ.uiuc.edu/~roger/research/ranks/ranks.pdf

# Plot the quantile regression
ggplot2::ggplot(data, aes(x , y)) +
  ggplot2::geom_point() + 
  ggplot2::geom_abline(intercept = coef(qr8)[1], slope = coef(qr8)[2])

# Let's test out a multiple quantile regression using the mtcars dataset
# MPG as a function of number of cylinders, engine volume (displacement), and
# vehicle weight regressing for the media.
qr <- quantreg::rq(mpg ~ cyl + disp + wt, data = mtcars, tau = 0.5)
summary(qr)

# Looks like in this additive model number of cylinders and weight are
# negatively related to median MPG (no big surprise), but engine volume doesn't
# have much/any explanatory power.

# This doesn't give me any estimate of statistical significance, but it seems
# there are other methods to estimate standard error that do
summary(qr, se = "iid")
summary(qr, se = "nid")
summary(qr, se = "boot")
summary(qr, se = "ker")