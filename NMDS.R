# Exploring NMDS using apple SNP data
# Reuses files derived in PCoA.R

# Clear environment
rm(list=ls())

# Set working directory and load libraries
setwd("~/Downloads")
library(vegan)
library(ggplot2)
library(stringr)

# Import apple accession phenotype metadata.
apple.p.metadata <- readxl::read_excel(
  "./20201102_meta_data.xlsx")

# Accessions have both a discovery and release year. Take the earliest year.
apple.p.metadata$Year <- apply(
  apple.p.metadata[, c("Discovered/described/cultivated", "Release Year")],
  1,
  function(x) ifelse(is.na(x[1]) & is.na(x[2]), NA, min(x, na.rm = TRUE)))
apple.p.metadata$Year <- as.numeric(apple.p.metadata$Year)

# Accessions have both a plant id and cultivar. Collate to one id preferencing
# plant id.
apple.p.metadata$Cultivar.New <- apply(
  apple.p.metadata[, c("PLANTID", "Cultivar")],
  1,
  function(x) ifelse(x[1] == "NA", x[2], x[1])
)

# Filter for only malus domestica accessions with a year, drop irrelevant
# columns, and rename cultivar variable.
apple.p.metadata <- apple.p.metadata[
  apple.p.metadata$species == "domestica" &
    !is.na(apple.p.metadata$Year),
  c("apple_id", "Cultivar.New", "Year", "Country", "Use", "type")]
names(apple.p.metadata)[
  names(apple.p.metadata) == "Cultivar.New"] <- "Cultivar"

# Filter for years since 1800 and add a column for time period. Time period
# is somewhat arbitrarily defined based on developmental stage of the apple
# industry.
apple.p.metadata <- apple.p.metadata[apple.p.metadata$Year >= 1800, ]
apple.p.metadata$Period <- apply(
  apple.p.metadata[, c("Year")],
  1,
  function(x) ifelse(
    x[1] >= 1800 & x[1] < 1900,
    1,
    ifelse(x[1] >= 1900 & x[1] < 1960, 2, 3))
)

# Load genetic distance data and IDs
dist <- read.table("./pcoa.mdist", header = F)
PID <- data.frame(
  PID = read.table("./pcoa.mdist.id", header = F)[, 1])
IID <- data.frame(
  IID = read.table("./pcoa.mdist.id", header = F)[, 2])

# Run NMDS for first 3 dimensions. Use Euclidean distance, no need to use a 
# different distance since input is just the hamming distance. Increasing
# default starts from 20 to 100 to help converge to lowest stress score. Not
# transforming data since isn't compositional and shouldn't be zero inflated.
# Could consider a transformation based on skewness, but leaving for now.
nmds <- vegan::metaMDS(
  dist,
  k = 3,
  distance = "euclidean",
  trymax = 100,
  autotransform = FALSE)

# Observe the stressplot. High degree of scatter away from the regression
# indicates a poor fit, but this looks okay.
vegan::stressplot(nmds)

# Stress of the best run was .18, which is not a great fit, but fair, usable.

# Construct dataframe to plot
results <- as.data.frame(vegan::scores(nmds))
results$apple_id <- as.double(
  stringr::str_sub(
    rownames(results),
    2,
    length(rownames(results))))
results <- dplyr::left_join(
  results,
  apple.p.metadata[, c("apple_id", "Period")],
  by = c("apple_id"))

# Graph first two axes
ggplot2::ggplot(data = results[!is.na(results$Period), ]) +
  ggplot2::geom_point(
    mapping = aes(x = NMDS1, y = NMDS2, color = as.factor(Period)),
    show.legend = TRUE) +
  ggplot2::scale_color_manual(
    "Period",
    values = c("1" = "yellow", "2" = "red", "3" = "blue")) +
  ggplot2::geom_hline(yintercept = 0, linetype = "dotted") +
  ggplot2::geom_vline(xintercept = 0, linetype = "dotted") +
  ggplot2::theme(legend.position = "top")

# Graph second two axes
ggplot2::ggplot(data = results[!is.na(results$Period), ]) +
  ggplot2::geom_point(
    mapping = aes(x = NMDS2, y = NMDS3, color = as.factor(Period)),
    show.legend = TRUE) +
  ggplot2::scale_color_manual(
    "Period",
    values = c("1" = "yellow", "2" = "red", "3" = "blue")) +
  ggplot2::geom_hline(yintercept = 0, linetype = "dotted") +
  ggplot2::geom_vline(xintercept = 0, linetype = "dotted") +
  ggplot2::theme(legend.position = "top")

# Similar result to PCoA, just based on visual inspection no real clustering
# between time periods.