# Exploring PCA using shark fin morphology data
# Data from https://datadryad.org/dataset/doi:10.5061/dryad.j3tx95x9b

# Clear environment
rm(list=ls())

# Set working directory and load libraries
setwd("~")
library(readxl)
library(dplyr)
library(ggplot2)

# Import data
data <- readxl::read_excel(
  "./Downloads/CompMorphSharkFins_dryad.xlsx",
  skip = 1)

# Delete second row because it's blank
data <- data[-1, ]

# Select certain continuous morphological variables to analyze using PCA
data.morph <- data[, c(
  "Species", "Fin area", "Fin length SL from base", "Fin length along LR",
  "Fin width base", "Fin width trailing", "Sk area", "Sk longest radial",
  "# radials")]

# Data contains different numbers of observations by species. Take the median
# value among individuals for a given species.
data.morph.agg <- data.morph %>%
  dplyr::group_by(Species) %>%
  dplyr::summarise(
    FinArea = median(`Fin area`),
    FinLengthSLFromBase = median(`Fin length SL from base`),
    FinLenghAlongLR = median(`Fin length along LR`),
    FinWidthBase = median(`Fin width base`),
    FinWidthTrailing = median(`Fin width trailing`),
    SkArea = median(`Sk area`),
    SkLongestRadial = median(`Sk longest radial`),
    NoRadials = median(`# radials`)
  )


# Run a principal component analysis on the data. Scale = TRUE to z-standardize
# since variables are in different units.
pca <- prcomp(
  subset(data.morph.agg, select = -c(Species)),
  scale = TRUE)

# Selecting which principal components to keep based on three criteria...

# Latent root criterion: drop components with eigenvalues < 1.
# Logic here is that such components represent less than one variable.
#
# Summarize PCA and square the standard deviations to get the eigenvalues.
pca.sum <- summary(pca)
pca.sum$sdev^2

# Takeaway: Keep only components 1 and 2 since the rest have eigenvalues < 1.

# Scree plot criterion: plot percentage of variance explained, keep components
# up until the "elbow," i.e. where the curve flattens.
# A bit subjective, but seems reasonable.
#
# Create data frame and plot
scree <- data.frame(
  PCA = colnames(pca.sum$importance),
  VarExplained = pca.sum$importance[2, ])
ggplot2::ggplot(scree, aes(y = VarExplained)) +
  ggplot2::geom_line(aes(x = PCA, group = 1))

# Takeaway: Definitely keep the first two components, arguably the 3rd too.

# Relative percent variance criterion: Keep components that cumuluatively
# explain at least 70% of the variance.
# Logic isn't really any different, again a bit subjective.
#
# Examine PCA summary
pca.sum$importance

# Takeaway: keep only the 1st component.

# Overall seems reasonable to examine the first two components.

# Examine loadings
pca

# First principal component is moderately positively correlated (< .4) with many
# dimensions that all relate to shark size (fin area, skeletal area, etc.) and
# uncorrelated to the number of radials.
# Second principal component is extremely positively correlated (> .9) with the
# number of radials and not particularly positively or negatively correlated
# with any other morphological features.
