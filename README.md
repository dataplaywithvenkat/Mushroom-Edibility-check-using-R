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

```

| Variable   | Description                                           | Data Type              |
|------------|-------------------------------------------------------|------------------------|
| Edible     | Indicates if a mushroom is edible or not              | Nominal, Categorical   |
| CapShape   | Provides the shape of the mushroom cap                | Nominal, Categorical   |
| CapSurface | Provides the surface texture of the mushroom cap      | Nominal, Categorical   |
| CapColor   | Provides the color of the mushroom cap                | Nominal, Categorical   |
| Odor       | Provides the odor or smell of the mushroom            | Nominal, Categorical   |
| Height     | Specifies if the mushroom is tall or short            | Nominal, Categorical   |


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
## Odor
Our data consists of 9 different Odor, but most of the mushrooms are without Odor, few mushrooms are with Musty ordor.


```r
table(data$Odor)
##
## Almond Anise Creosote Fishy Foul Musty None Pungent
## 400 400 192 576 2160 36 3528 256
## Spicy
## 576

ggplot(data, aes(Odor))+geom_bar(fill = "#0073C2FF")

```
## Height
It consists of two items namely short and tall, they both are having nearly equal split in our Data.

```r
table(data$Height)
##
## Short Tall
## 4043 4081

ggplot(data, aes(Height))+geom_bar(fill = "#0073C2FF")

```
# Data Preprossing
As our data is a categorical we can’t use it for analysis we need to convert the categorical data to factor as.factor() will help to convert
categorical data to factor

```r
data$Edible = as.factor(data$Edible)
data$CapShape = as.factor(data$CapShape)
data$CapSurface = as.factor(data$CapSurface)
data$CapColor = as.factor(data$CapColor)
data$Odor = as.factor(data$Odor)
data$Height = as.factor(data$Height)
str(data)


## 'data.frame': 8124 obs. of 6 variables:
## $ Edible : Factor w/ 2 levels "Edible","Poisonous": 2 1 1 2 1 1 1 1 2 1 ...
## $ CapShape : Factor w/ 6 levels "Bell","Conical",..: 3 3 1 3 3 3 1 1 3 1 ...
## $ CapSurface: Factor w/ 4 levels "Fibrous","Grooves",..: 4 4 4 3 4 3 4 3 3 4 ...
## $ CapColor : Factor w/ 10 levels "Brown","Buff",..: 1 10 9 9 4 10 9 9 9 10 ...
## $ Odor : Factor w/ 9 levels "Almond","Anise",..: 8 1 2 8 7 1 1 2 8 1 ...
## $ Height : Factor w/ 2 levels "Short","Tall": 2 1 2 1 1 1 1 2 2 2 ...

```
After using the factor function we can convert the data to Train dataset and test dataset. we are using Edible as our traget variable based on our business problem.

``` r
train_index <- createDataPartition(data$Edible, p = 0.7, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]
print(dim(train_data))

## [1] 5688 6

print(dim(test_data))
## [1] 2436 6
```
from above code we can observe that dataset is split having 70% as training set and 30% is the testing set.
We need to perform modelling for based on using few different columns below code will helps us to give different formulas as a input in here are
are considering the Edible as a Traget variables and other columns as input variable.
`Edible ~ CapColor` : for this formula we are using Edible as Target and CapColor as Input
`Edible ~ CapSurface` : We are using only CapSurface as Input
`Edible ~ CapShape` : We are using Capshape as Input
`Edible ~ Odor` : We are using Odor as Input
`Edible ~ CapSurface+CapColor+CapShape` : We are using CapSurface, CapColor, CapShape as Input variables
`Edible ~ CapSurface+CapColor+CapShape+Height` : We are using CapSurface, CapColor, CapShape and Height as Input variables
`Edible ~ CapSurface+CapColor+CapShape+Odor` : We are using CapSurface, CapColor, CapShape and Odor as Input variables
`Edible ~ .` : We are using All other columns expect Edible as the Input variables.

```r
x <- Edible ~ .
x1 <- Edible ~ Odor
x2 <- Edible ~ CapSurface+CapColor+CapShape
x3 <- Edible ~ CapSurface+CapColor+CapShape+Odor
x4 <- Edible ~ CapColor
x5 <- Edible ~ CapSurface
x6 <- Edible ~ CapShape
x7 <- Edible ~ CapSurface+CapColor+CapShape+Height
y <- c(x4,x5,x6,x7,x2,x1,x3,x)
```

## Logistic Regression

### Algorithm:

1. **Initializing for loop**: To perform the model with different formulas.
2. **Training the model**: Using the `glm()` function with `Train_Data`. As the target variable has two values (Edible or Poisonous), we use the family as binomial, which converts the normal model to a logistic regression model.
3. **Predicting the model**: On the `Test_data`, with type as response.
4. **Converting predicted values**: Since the predicted values are in numbers, converting them into Edible or Poisonous by keeping the threshold as 0.5. Prediction values < 0.5 are Edible and > 0.5 are Poisonous as per the method of logistic regression.
5. **Finding the number of correctly predicted values, accuracy**: Initializing it to a list for further analysis.

```r
my_list_log <- c()
original_log <- c()
accuracy_log <- c()
for (i in y){
 model = glm(i, data = train_data, family = binomial)
 predictions <- predict(model, newdata = test_data, type = 'response')
 predictions <- as.factor(ifelse(predictions < 0.5, 'Edible', 'Poisonous'))
 my_list_log <- append(my_list_log,sum(test_data$Edible==predictions))
 original_log <- append(original_log,length(test_data$Edible))
 accuracy_log <- append(accuracy_log,(sum(test_data$Edible == predictions) / length(test_data$Edible)))
}
```
## Output:
Below code give the output of logistic regression algorithm in tabular form using Knitr varaiable and Dataframe.

```r
x <- "Edible ~ ."
x1 <- "Edible ~ Odor"
x2 <- "Edible ~ CapSurface+CapColor+CapShape"
x3 <- "Edible ~ CapSurface+CapColor+CapShape+Odor"
x4 <- "Edible ~ CapColor"
x5 <- "Edible ~ CapSurface"
x6 <- "Edible ~ CapShape"
x7 <- "Edible ~ CapSurface+CapColor+CapShape+Height"
# Create a data frame with the formulas and predictions
z <- data.frame(Formula = c(x4,x5,x6,x7,x2,x1,x3,x),
 Test_data = original_log,
 Predictions = my_list_log,
 Accuracy = accuracy_log*100)
# Print the table
kable(z, caption = "Table 1: Prediction and Accuracy for respective formulas in logistic Regression")
```
## Table 1: Prediction and Accuracy for respective formulas in Logistic Regression

| Formula                                       | Test_data | Predictions | Accuracy   |
|-----------------------------------------------|-----------|-------------|------------|
| Edible ~ CapColor                            | 2436      | 1470        | 60.34483   |
| Edible ~ CapSurface                          | 2436      | 1436        | 58.94910   |
| Edible ~ CapShape                            | 2436      | 1377        | 56.52709   |
| Edible ~ CapSurface+CapColor+CapShape+Height | 2436      | 1638        | 67.24138   |
| Edible ~ CapSurface+CapColor+CapShape        | 2436      | 1638        | 67.24138   |
| Edible ~ Odor                                | 2436      | 2395        | 98.31691   |
| Edible ~ CapSurface+CapColor+CapShape+Odor   | 2436      | 2415        | 99.13793   |
| Edible ~ .                                   | 2436      | 2415        | 99.13793   |

From above table we can observe that the model with only CapColor, CapSurface, Capshape and all the combined are giving less accuracy when
compared with the formula with Order as the input it is giving 98.8 accuracy, Using all the columns as input the Edible column is correctly predicted
with 99.17% of accuracy.






