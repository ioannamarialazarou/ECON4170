# load the required R libraries (install the respective packages if needed)
library(mlbench)
library(caret)
library(tidyverse)
library(dplyr)
library(Boruta)

library(Metrics)
library(pROC)

source('classification_magic.R')

# Read data (comma separated)
data = read.table('magic04.data',sep = ',')

# Rename columns 
colnames(data) = c('fLength','fWidth','fSize','fConc','fConc1','fAsym','fM3Long','fM3Trans','fAlpha','fDist','class')

# Check for missing (NA) values 
sapply(data, function(x) sum(is.na(x)))
# No missing values for any of the features involved
# We assume that all reported values for each of the features are within the valid range

# For testing purposes (Pick 500 observation in a random fashion)
data = data[sample(nrow(data), 500), ]

classification_magic(data, fs = 2, balance = 3, classifier = 2, tune = 1)


