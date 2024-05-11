# Statistical Learning Accessible Practical

## Introduction
This repository contains practical implementations of statistical learning techniques using R. The aim is to provide accessible examples for learning purposes.

## Necessary Libraries
To run the code in this repository, the following R libraries are necessary:

1. **dplyr**: Used for basic data preprocessing.
2. **ggplot2**: Utilized for displaying graphs.
3. **caret**: Enables the creation of data partitions, training and testing sets, and confusion matrices.
4. **rpart**: Required for building decision tree models.
5. **rpart.plot**: Used for visualizing decision tree models.
6. **randomForest**: Necessary for implementing random forest models.
7. **knitr**: Used to visualize outputs in the form of tables.

## Data Characterization

### Number of Variables and Types
Let's examine the number of variables and their types in the dataset:

```R
names(data)
## [1] "Edible" "CapShape" "CapSurface" "CapColor" "Odor" "Height"

str(data)
## 'data.frame': 8124 obs. of 6 variables:
##  $ Edible    : chr "Poisonous" "Edible" "Edible" "Poisonous" ...
##  $ CapShape  : chr "Convex" "Convex" "Bell" "Convex" ...
##  $ CapSurface: chr "Smooth" "Smooth" "Smooth" "Scaly" ...
##  $ CapColor  : chr "Brown" "Yellow" "White" "White" ...
##  $ Odor      : chr "Pungent" "Almond" "Anise" "Pungent" ...
##  $ Height    : chr "Tall" "Short" "Tall" "Short" ...

