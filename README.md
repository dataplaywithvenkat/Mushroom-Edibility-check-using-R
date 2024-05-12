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
```
```r
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
```

```r
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
We need to perform modelling for based on using few different columns below code will helps us to give different formulas as a input in here are considering the Edible as a Traget variables and other columns as input variable.
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

## Cross validation:
Once we have selected the formula for our model we need to fine tune to get more accurate results and perform cross validation to get justify if our
model is good or not. based on the above table I am including only one formula Edible ~ . Selecting all the columns as Input except Edible as
output.

Algorithm:
1. initialize the Split of train data and test data. I have used 50% TO 90% of splits
2. Running above logistic regression algorithm for single formula Edible ~ .
3. finding the number of correctly predicted values and accuracy for each split.

```r
train_split = c(0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9)
log_accuracy_split = c()
original_train_log = c()
original_test_log = c()
my_list_split_log = c()
for (i in train_split){
 train_index_split <- createDataPartition(data$Edible, p = i, list = FALSE)
 train_data_split <- data[train_index_split, ]
 test_data_split <- data[-train_index_split, ]
 log_model <- glm(Edible ~ ., data = train_data_split, family = binomial)
 predictions <- predict(model, newdata = test_data_split, type = 'response')
 predictions <- as.factor(ifelse(predictions < 0.5, 'Edible', 'Poisonous'))
 my_list_split_log <- append(my_list_split_log,sum(test_data_split$Edible==predictions))
 original_train_log <- append(original_train_log,length(train_data_split$Edible))
 original_test_log <- append(original_test_log,length(test_data_split$Edible))
 log_accuracy_split <- append(log_accuracy_split,(sum(test_data_split$Edible == predictions) / length(test_data_split$Edibl
e)))
}
```
```r
# Create a data frame with the formulas and predictions
z <- data.frame(train_split = train_split,
 Train_Data = original_train_log,
 Test_data = original_test_log,
 Predictions = my_list_split_log,
 Accuracy = log_accuracy_split*100)
# Print the table
kable(z,caption = "Table 2: Predictions in Logistic regression for different percentage of splits")
```
Table 2: Predictions in Logistic regression for different percentage of splits

| train_split | Train_Data | Test_data | Predictions | Accuracy  |
|-------------|------------|-----------|-------------|-----------|
| 0.50        | 4062       | 4062      | 4032        | 99.26145  |
| 0.55        | 4469       | 3655      | 3626        | 99.20657  |
| 0.60        | 4875       | 3249      | 3230        | 99.41520  |
| 0.65        | 5282       | 2842      | 2820        | 99.22590  |
| 0.70        | 5688       | 2436      | 2417        | 99.22003  |
| 0.75        | 6093       | 2031      | 2016        | 99.26145  |
| 0.80        | 6500       | 1624      | 1615        | 99.44581  |
| 0.85        | 6906       | 1218      | 1205        | 98.93268  |
| 0.90        | 7313       | 811       | 806         | 99.38348  |

We can observe from the above table that the logistic regression is providing the average accuracy of 99.3 with train and test split, which is a good understanding that the model works well for Mushroom detection.

Decision Tree
for the decision tree model we need to first plot the tree node splits, below code we helps us to create plot for different formulas and shows there
importance:
```r
mytree1 = rpart(Edible ~ Odor, data = train_data, method = 'class')
rpart.plot(mytree1)
````

The above plot displays that when we are using the default complexity parameter values and Odor as only input variable we can find the node split
is happening once
```r
mytree1 = rpart(Edible ~ CapShape+CapColor+CapSurface, data = train_data, method = 'class')
rpart.plot(mytree1)
```

The above plot displays that when we are using the default complexity parameter values and CapShape, CapColor, CapSurface , Odor as input
variable we can find the node split is using only single variable, this can be happening because of below reasons:
1. Variable importance of Odor is more compared to other variables
2. Complexity parameter is the default value.


The code give the variable importance of each variable using MeanGini Impurity value for our Data in decreasing order.
```r
model <- rpart(Edible ~ ., data = data, method = 'class') #Decision Tree model
variable_importance <- function(model) {
 imp <- model$variable.importance
 imp <- imp / max(imp) # Normalize importance scores to the maximum value
 imp <- sort(imp, decreasing = TRUE) # Sort in descending order
 return(imp)
}
# Get variable importance scores based on Gini impurity
importance_scores <- variable_importance(model)
# Plot variable importance based on Gini impurity
barplot(importance_scores, main = "Variable Importance Plot", xlab = "Variables", ylab = "Mean Gini Impurity")

```
From above bar we can observe that the variable importance of Odor is nearly equal to one while others is nearly 0.2, this can be said that Gini impurity of Odor is highest. this is reason why data is only spitted based of Odor variable

Below code displays how complexity parameter value can change use of variables for prediction.

```r
cp_v = c(0.1,0.01,0.000001,0.000002)
for (i in cp_v){
 model <- rpart(Edible ~ ., data = train_data,cp = i)
 predictions <- predict(model, newdata = test_data, type = "class")
# Print the CP table
 cat("\n")
 cat("cp_value used for Decision tree",i,"\n")
 printcp(model)
}
```

This can be visualized by below R plot.
```r
mytree1 = rpart(Edible ~ CapShape+CapColor+CapSurface+Odor, data = train_data, method = 'class',cp = 0.0000001)
rpart.plot(mytree1)
```

Compared to all the above plots this graph is using its all the variables in predicting the values, this happens due to the complexity score as 0.0000001
Algorithm:
1. Initializing for loop to perform model with different formulas, and complexity parameter as 0.0000001
2. Training the model with rpart() funtion with Train_Data, and with all the above formulas as specified.
3. Predicting the model on the Test_data, with type as class.
4. finding the number of correctly predicted values, accuracy and initializing it to list for further analysis.

```r
my_list_dt <- c()
original_dt <- c()
accuracy_dt <- c()
for (i in y){
 model = rpart(i, data = train_data,cp=0.0000001)
 predictions <- predict(model, newdata = test_data, type = "class")
 my_list_dt <- append(my_list_dt,sum(test_data$Edible==predictions))
 original_dt <- append(original_dt,length(test_data$Edible))
 accuracy_dt <- append(accuracy_dt,(sum(test_data$Edible == predictions) / length(test_data$Edible)))
}
```
Output:
Below code give the output of Decision tree algorithm in tabular form using Knitr varaiable and Dataframe.

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
 Test_data = original_dt,
 Predictions = my_list_dt,
 Accuracy = accuracy_dt*100)
# Print the table
kable(z,caption = "Table 3: Prediction and Accuracy Decision tree with different formulas")
```

Table 3: Prediction and Accuracy Decision tree with different formulas

| Formula                                | Test_data | Predictions | Accuracy   |
|----------------------------------------|-----------|-------------|------------|
| Edible ~ CapColor                     | 2436      | 1470        | 60.34483   |
| Edible ~ CapSurface                   | 2436      | 1436        | 58.94910   |
| Edible ~ CapShape                     | 2436      | 1377        | 56.52709   |
| Edible ~ CapSurface+CapColor+CapShape+Height | 2436 | 1711        | 70.23810   |
| Edible ~ CapSurface+CapColor+CapShape | 2436      | 1718        | 70.52545   |
| Edible ~ Odor                         | 2436      | 2395        | 98.31691   |
| Edible ~ CapSurface+CapColor+CapShape+Odor | 2436 | 2412        | 99.01478   |
| Edible ~ .                            | 2436      | 2412        | 99.01478   |

From above table we can observe that the model with only CapColor, CapSurface, Capshape and all the combined are giving less accuracy when
compared with the formula with Order as the input it is giving 98.8 accuracy, Using all the columns as input the Edible column is correctly predicted
with 99.8% of accuracy.

## Cross validation:
We are performing the cross validation using different test and train split on the decision tree model from the below code.

```r
train_split = c(0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9)
dt_accuracy_split = c()
original_train_dt_split = c()
original_test_dt_split = c()
my_list_split_dt = c()
for (i in train_split){
 train_index_dt <- createDataPartition(data$Edible, p = i, list = FALSE)

 train_data_split <- data[train_index_dt, ]

 test_data_split <- data[-train_index_dt, ]

 model = rpart(Edible ~ CapSurface+CapColor+CapShape+Odor, data = train_data_split)

 predictions <- predict(model, newdata = test_data_split, type = "class")

 my_list_split_dt <- append(my_list_split_dt,sum(test_data_split$Edible==predictions))

 original_train_dt_split <- append(original_train_dt_split,length(train_data_split$Edible))

 original_test_dt_split <- append(original_test_dt_split,length(test_data_split$Edible))

 dt_accuracy_split <- append(dt_accuracy_split,(sum(test_data_split$Edible == predictions) / length(test_data_split$Edibl
e)))

}
```

Output
Output is shown from the below code for respective test and train splits.
# Create a data frame with the formulas and predictions

```r
z <- data.frame(train_split = train_split,
 Train_Data = original_train_dt_split,
 Test_data = original_test_dt_split,
 Predictions = my_list_split_dt,
 Accuracy = dt_accuracy_split*100)
# Print the table
kable(z,caption = "Table 4: Prediction and Accuracy Decision tree with different splits")
Table 4: Prediction and Accuracy Decision tree with different splits
```

Table: Prediction and Accuracy Decision tree with different splits

| train_split | Train_Data | Test_data | Predictions | Accuracy  |
|-------------|------------|-----------|-------------|-----------|
| 0.50        | 4062       | 4062      | 4013        | 98.79370  |
| 0.55        | 4469       | 3655      | 3600        | 98.49521  |
| 0.60        | 4875       | 3249      | 3203        | 98.58418  |
| 0.65        | 5282       | 2842      | 2794        | 98.31105  |
| 0.70        | 5688       | 2436      | 2401        | 98.56322  |
| 0.75        | 6093       | 2031      | 2014        | 99.16297  |
| 0.80        | 6500       | 1624      | 1606        | 98.89163  |
| 0.85        | 6906       | 1218      | 1195        | 98.11166  |
| 0.90        | 7313       | 811       | 799         | 98.52035  |

from above output we can observe that the average accuracy for Decision tree is 98.8 for formula `Edible~.`

# Random Forest
Initially, we are ploting the Random forest model with our train data with number of trees as 500 to get an assumption and left the mtry as the
default.
```r
model <- randomForest(Edible ~., data = train_data, ntree = 500)
plot(model, col = c("blue", "green", "red"))
```
 from above graph we can observe
that the error is getting decreased from number of trees as 100 I am using the number of trees as 100 as per the above graph once iot is done we
need to consider the variable importance using Gini Impurity
Below code plots the Variable importance using `varImpPlot()` funtion
```r
model <- randomForest(Edible ~ ., data = train_data, ntree = 100)
# Create the variable importance plot
varImpPlot(model)
```

The model is also considering the
same Gini impurity with Odor as the heighest, Height as the lowest
Algorithm:
1. Initializing for loop to perform model with different formulas
2. Training the model with randomForest() funtion with Train_Data, and with all the above formulas as specified, specifying ntree, mtry.
3. Predicting the model on the Test_data, with type as class.
4. finding the number of correctly predicted values, accuracy and initializing it to list for further analysis.
```r
rf_my_list <- c()
rf_original <- c()
rf_accuracy <- c()
for (i in y){
 model = randomForest(i, data = train_data, ntree = 100)
 predictions <- predict(model, newdata = test_data, type = "class")
 rf_my_list <- append(rf_my_list,sum(test_data$Edible==predictions))
 rf_original <- append(rf_original,length(test_data$Edible))
 rf_accuracy <- append(rf_accuracy,(sum(test_data$Edible == predictions) / length(test_data$Edible)))
}
```
Output:
Below code give the output of Random Forest algorithm in tabular form using Knitr varaiable and Dataframe.
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
 Test_data = rf_original,
 Predictions = rf_my_list,
 Accuracy = rf_accuracy*100)

# Print the table
kable(z,format = "simple",caption = "Table 5: Prediction and Accuracy Random forest with different Formulas")
```
Table 5: Prediction and Accuracy Random forest with different Formulas










