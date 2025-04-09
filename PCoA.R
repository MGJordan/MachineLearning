# Exploring PCoA using apple SNP data
# SNP data from https://datadryad.org/dataset/doi:10.5061/dryad.zkh1893cd
# Apple cultivar metadata from:
# https://github.com/MylesLab/abc-phenomics/tree/main/data
# Uses Plink v1.90 to calculate genetic distance:
# https://www.cog-genomics.org/plink2/

# Clear environment
rm(list=ls())

# Set working directory and load libraries
setwd("~/Downloads")
library(readxl)
library(ggplot2)

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

# Use Plink to remove SNPs in linkage disequilibrium. This is done because
# these SNPs are highly correlated and so contain redundant information and are
# not independent.
system("/usr/local/bin/Plink/plink --file abc_combined_maf001_sort_vineland_imputed --indep-pairwise 10 3 0.5")

# Get the genetic distance between accessions using Plink (this is for all
# periods combined)
# This will be the hamming distance, i.e. the percentage of SNPs that are the
# same.
system("/usr/local/bin/Plink/plink --file abc_combined_maf001_sort_vineland_imputed --extract plink.prune.in --allow-no-sex --geno .05 --mind .1 --maf .01 --distance-matrix --out ./pcoa")

# Load genetic distance data and IDs
dist <- read.table("./pcoa.mdist", header = F)
PID <- data.frame(
  PID = read.table("./pcoa.mdist.id", header = F)[, 1])
IID <- data.frame(
  IID = read.table("./pcoa.mdist.id", header = F)[, 2])

# Perform PCoA, returning eigenvalue for first 5 dimensions.
mds <- cmdscale(dist, eig = T, k = 5)

# Extract eigen vectors and bind with IDs and period
eigenvec <- cbind(PID, IID, mds$points)
map2 <- apple.p.metadata[, c("apple_id", "Period")]
eigenvec <- dplyr::left_join(eigenvec, map2, by = join_by(IID == apple_id))

# Calculate proportion of variation captured by each eigenvector
eigen.perc <- round(((mds$eig) / sum(mds$eig)) * 100, 2)

# Graph first two axes
ggplot2::ggplot(data = eigenvec[!is.na(eigenvec$Period), ]) +
  ggplot2::geom_point(
    mapping = aes(x = `1`, y = `2`, color = as.factor(Period)),
    show.legend = TRUE) +
  ggplot2::scale_color_manual(
    "Period",
    values = c("1" = "yellow", "2" = "red", "3" = "blue")) +
  ggplot2::geom_hline(yintercept = 0, linetype = "dotted") +
  ggplot2::geom_vline(xintercept = 0, linetype = "dotted") +
  ggplot2::labs(
    title = "PCoA of apple cultivar SNPs",
    x = paste0("PCoA Axis 1 (", eigen.perc[1], " %)"),
    y = paste0("PCoA Axis 2 (", eigen.perc[2], " %)")) +
  ggplot2::theme(legend.position = "top")

# Looks like they don't explain much of the variation (~15%). Based on visual
# inspection cultivars don't seem to vary in terms of genetic distance by time
# period.

# Graph axes two and three
ggplot2::ggplot(data = eigenvec[!is.na(eigenvec$Period), ]) +
  ggplot2::geom_point(
    mapping = aes(x = `2`, y = `3`, color = as.factor(Period)),
    show.legend = TRUE) +
  ggplot2::scale_color_manual(
    "Period",
    values = c("1" = "yellow", "2" = "red", "3" = "blue")) +
  ggplot2::geom_hline(yintercept = 0, linetype = "dotted") +
  ggplot2::geom_vline(xintercept = 0, linetype = "dotted") +
  ggplot2::labs(
    title = "PCoA of apple cultivar SNPs",
    x = paste0("PCoA Axis 2 (", eigen.perc[2], " %)"),
    y = paste0("PCoA Axis 3 (", eigen.perc[3], " %)")) +
  ggplot2::theme(legend.position = "top")

# These axes explain ~9% of the data. Overall PCoA doesn't seem very informative
# given the low percentage of variation in the data explained by the primary
# axes.