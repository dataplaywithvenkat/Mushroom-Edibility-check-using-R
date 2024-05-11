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

| Variable   | Description                                           | Data Type              |
|------------|-------------------------------------------------------|------------------------|
| Edible     | Indicates if a mushroom is edible or not              | Nominal, Categorical   |
| CapShape   | Provides the shape of the mushroom cap                | Nominal, Categorical   |
| CapSurface | Provides the surface texture of the mushroom cap      | Nominal, Categorical   |
| CapColor   | Provides the color of the mushroom cap                | Nominal, Categorical   |
| Odor       | Provides the odor or smell of the mushroom            | Nominal, Categorical   |
| Height     | Specifies if the mushroom is tall or short            | Nominal, Categorical   |

```
## Exploratory Data Analysis

### Edible

The variable "Edible" consists of two main categories: Edible or Poisonous, with 4208 rows labeled as Edible and 3916 rows labeled as Poisonous.

```r
print(table(data$Edible))
##
## Edible    Poisonous
##  4208       3916

ggplot(data, aes(Edible))+geom_bar(fill = "#0073C2FF")
```
## CapShape

The variable "CapShape" consists of 6 main categories: bell, conical, convex, flat, knobbed, and sunken. Among these, convex and flat are the most occurring cap shapes, while conical and sunken are the least occurring.

```r
table(data$CapShape)
##
## Bell   Conical   Convex   Flat   Knobbed   Sunken
##  452        4      3656   3152      828       32

ggplot(data, aes(CapShape))+geom_bar(fill = "#0073C2FF") 
```

## CapColor

The variable "CapColor" consists of 10 different colors such as Brown, Buff, Cinnamon, etc. Most mushrooms are found to be brown or gray in color, while mushrooms with green and purple colors are very rare.

```r
table(data$CapColor)
##
## Brown  Buff Cinnamon  Gray Green  Pink Purple   Red White Yellow
##  2284   168      44  1840    16   144     16  1500  1040   1072

ggplot(data, aes(CapColor)) + geom_bar(fill = "#0073C2FF")

```

## CapSurface

The variable "CapSurface" consists of 4 cap surface types: fibrous, grooves, scaly, and smooth. Mushrooms with a scaly surface are more common, while those with grooves are less common.

```r
table(data$CapSurface)
##
## Fibrous  Grooves   Scaly  Smooth
##    2320        4    3244    2556

ggplot(data, aes(CapSurface))+geom_bar(fill = "#0073C2FF") 

```



