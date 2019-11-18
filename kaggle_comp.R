install.packages("tidyverse")
install.packages("corrplot")
install.packages("caret")
install.packages("dplR")
install.packages("caTools")
install.packages("broom")
install.packages("MLmetrics")
install.packages("neuralnet")
install.packages("xgboost")
#install.packages("MASS")
#install.packages("topicmodels")
library(tidyverse)
library(corrplot)
library(caret)
library(dplR)
library(caTools)
library(broom)
library(randomForest)
library(rpart)
library(modelr)
library(MLmetrics)
library(neuralnet)
library(xgboost)
#library(MASS)
#library(tm)


#loading data

df <- read.csv("vw_kaggle_train.csv", stringsAsFactors = F)
df <- filter(df,!my_label=="")
dim(df)
head(df)
summary(df)
str(df)

test <- read.csv("vw_kaggle_test.csv", stringsAsFactors = F)
dim(test)
head(test)
View(test)

sample <- read.csv("sample_answer.csv", stringsAsFactors = F)
head(sample)
View(sample)

#checking feature correlation 

grep("hertz", colnames(df)) #checking index of hertz
df <- select(df,-c(10)) #removing Hz from df to make correlation analysis easier 
grep("timestamp", colnames(df))
df <- select(df,-c(29))
grep("id", colnames(df))
df <- select(df,-c(1))

feature_correlations1 <- cor(df[1:27])
print(feature_correlations1)
corrplot(feature_correlations1,method="circle")
feature_highcorrelation <- findCorrelation(feature_correlations1, cutoff=0.75) #find highly correlated features, generally should remove reatures with greater than 75% correlation
print(feature_highcorrelation)
colnames(df[c(6,14,12,13,10,15,9,11,5,25,27,23,22,21,24,17)]) #these indexes represent variables with absolutle correlation above 75%. Can remove from training data

df_clean <- select(df,-c(6,14,12,13,10,15,9,11,5,25,27,23,22,21,24,17))
dim(df_clean)
head(df_clean)

#convert labels to factor 
str(df_clean$my_label)
df_clean$my_label <- as.factor(df_clean$my_label)
levels(df_clean$my_label)
levels(df_clean$my_label)[1]

#removing levels with occurance <4 **not for K-folds
summary(df_clean$my_label)
df_clean_split <- df_clean
str(df_clean_split)
df_clean_split <- filter(df_clean_split,!my_label=="+kettle")
df_clean_split <- filter(df_clean_split,!my_label=="+shower")
df_clean_split <- filter(df_clean_split,!my_label=="+fridge+vacuum+washer_dryer")
df_clean_split <- filter(df_clean_split,!my_label=="+fridge+kettle+tumble_dryer+washer_dryer+microwave")
df_clean_split <- filter(df_clean_split,!my_label=="+fridge+shower+kettle")
df_clean_split$my_label <- droplevels(df_clean_split$my_label)
summary(df_clean_split$my_label)

#holding back 25% data for testing
#split train and test data 

set.seed(345)
sample = sample.split(df_clean, SplitRatio = .75)
train1 = subset(df_clean, sample == TRUE)
test1  = subset(df_clean, sample == FALSE)
dim(train1)
head(train1)
dim(test1)

str(train1)
summary(train1$my_label)
str(test1)
summary(test1$my_label)


# LDA ---------------------------------------------------------------------

#training LDA model
lda1 <- lda(my_label ~ type+activePower+activePowerDelta+reactivePower+
             voltage+phase+transient8+transient10+harmonicDelta1+harmonicDelta2+
             harmonicDelta8, data=train1)

predictions_lda <- predict(lda1, test1, type = "class")
print(predictions_lda)
confusionMatrix(predictions_lda, test1$my_label)
F1_lda <- F1_Score(y_true = test1$my_label, y_pred = predictions_lda, positive = NULL)


# SINGLE TREE -------------------------------------------------------------

set.seed(456)
dt <- rpart(my_label ~ type+activePower+activePowerDelta+reactivePower+
              voltage+phase+transient8+transient10+harmonicDelta1+harmonicDelta2+
              harmonicDelta8, data=train1, method = "class")

print(dt)

predictions_dt <- predict(dt, test1, type = "class")
confusionMatrix(predictions_dt, test1$my_label)
F1_DT1 <- F1_Score(y_true = test1$my_label, y_pred = predictions_dt , positive = NULL)
F1_DT1


# RANDOM FOREST -----------------------------------------------------------

#train1$my_label <- droplevels(train1$my_label)
#test1$my_label <- droplevels(test1$my_label)

set.seed(456)
forest1 <- randomForest(my_label ~ type + activePower+activePowerDelta+reactivePower+
                          voltage+phase+transient8+transient10+harmonicDelta1+harmonicDelta2+
                          harmonicDelta8, data=train1, importance = TRUE, ntree = 1000)

predictions_rf <- predict(forest1, test1, type = "class")
confusionMatrix(predictions_rf, test1$my_label)
F1_RF1 <- F1_Score(y_true = test1$my_label, y_pred = predictions_rf, positive = NULL)
F1_RF1


# NEURAL NETWORK ----------------------------------------------------------

#normalising data 
scale_train1 <- scale(train1[c(1:11)])
scale_train1 <- cbind(scale_train1, train1[!names(train1) %in% names(scale_train1)])

scale_test1 <- scale(test1[c(1:11)])
scale_test1 <- cbind(scale_test1, test1[!names(test1) %in% names(scale_test1)])

NN = neuralnet(class ~ type + activePower+activePowerDelta+reactivePower+
                 voltage+phase+transient8+transient10+harmonicDelta1+harmonicDelta2+
                 harmonicDelta8, data = scale_train1, hidden = 11 , stepmax = 1e+10, linear.output = F )


# XG BOOST ----------------------------------------------------------------


#adding new variable to store labels for future access
levels <- levels(df_clean$my_label)

# Convert my_label factor to integers
df_clean_boost <- df_clean
df_clean_boost$class <- as.integer(df_clean$my_label)-1
label <- df_clean_boost$class
label <- as.data.frame(label)#label vector for XGBOOT matrix 

df_clean_boost$my_label = NULL
df_clean_boost$class = NULL #feature variables for XGBOOST matrix 


#splitting data to train and test
set.seed(123)
sample = sample.split((df_clean), SplitRatio = .75)
train_data = subset(df_clean_boost, sample == TRUE)
train_label = subset(label, sample ==TRUE)
test_data  = subset(df_clean_boost, sample == FALSE)
test_label = subset(label, sample == FALSE)

train_label <- as.matrix(train_label)
train_data <- as.matrix(train_data)
test_label <- as.matrix(test_label)
test_data <- as.matrix(test_data)

dim(train_data)
dim(train_label)
dim(test_data)
dim(test_label)

#create the xgb.DMatrix objects
xgb_train <- xgb.DMatrix(data=train_data,label=train_label)
xgb_test <- xgb.DMatrix(data=test_data,label=test_label)

#setting params
num_class = length(levels)
params = list(
  booster="gbtree",
  eta=0.001,
  max_depth=5,
  gamma=3,
  subsample=0.75,
  colsample_bytree=1,
  objective="multi:softprob",
  eval_metric="mlogloss",
  num_class=num_class
)

#dtraining XB
xgb_fit <- xgb.train(
            params=params,
            data=xgb_train,
            nrounds=500,
            nthreads=1,
            early_stopping_rounds=5,
            watchlist=list(val1=xgb_train,val2=xgb_test),
            verbose=1
            )

xgb_fit

#predicting new values 

xgb_pred <- predict(xgb_fit,test_data,reshape=T)
xgb_pred <- as.data.frame(xgb_pred)
colnames(xgb_pred) = levels(df_clean$my_label)

xgb_pred$prediction <- apply(xgb_pred,1,function(x) colnames(xgb_pred)[which.max(x)])
xgb_pred$label <- levels(df_clean$my_label)[test_label+1]
CM <- confusionMatrix(as.factor(xgb_pred$prediction), as.factor(xgb_pred$label))
F1_RF1 <- F1_Score(y_true = xgb_pred$label, y_pred = xgb_pred$prediction, positive = NULL)
F1_RF1



# Classification with k-folds ---------------------------------------------







