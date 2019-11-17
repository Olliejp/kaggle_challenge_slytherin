install.packages("tidyverse")
install.packages("corrplot")
install.packages("caret")
install.packages("dplR")
install.packages("caTools")
install.packages("broom")
install.packages("MLmetrics")
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


#loading data

df <- read.csv("vw_kaggle_train.csv", stringsAsFactors = F)
#df$my_label <- as.factor(df$my_label) - can do this later if needed 
dim(df)
head(df)
summary(df)
str(df)

df <- filter(df,!my_label=="")

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

str(df_clean$my_label)
df_clean$my_label <- as.factor(df_clean$my_label)
levels(df_clean$my_label)
levels(df_clean$my_label)[1]

#holding back 25% data for testing
#split train and test data 
set.seed(123)
sample = sample.split(df_clean, SplitRatio = .75)
train1 = subset(df_clean, sample == TRUE)
test1  = subset(df_clean, sample == FALSE)
dim(train1)
head(train1)
dim(test1)

str(train1)
str(test1)
#train1$my_label <- droplevels(train1$my_label)
#test1$my_label <- droplevels(test1$my_label)

#training DT
set.seed(456)
dt <- rpart(my_label ~ type+activePower+activePowerDelta+reactivePower+
              voltage+phase+transient8+transient10+harmonicDelta1+harmonicDelta2+
              harmonicDelta8, data=train1, method = "class")

print(dt)

predictions_dt <- predict(dt, test1, type = "class")
confusionMatrix(predictions_dt, test1$my_label)
F1_DT1 <- F1_Score(y_true = test1$my_label, y_pred = predictions_dt , positive = NULL)
F1_DT1

#training a random forest 

train1$my_label <- droplevels(train1$my_label)
#test1$my_label <- droplevels(test1$my_label)

set.seed(456)
forest1 <- randomForest(my_label ~ type + activePower+activePowerDelta+reactivePower+
                          voltage+phase+transient8+transient10+harmonicDelta1+harmonicDelta2+
                          harmonicDelta8, data=train1, importance = TRUE, ntree = 1000)

predictions_rf <- predict(forest1, test1, type = "class")
confusionMatrix(predictions_rf, test1$my_label)
F1_RF1 <- F1_Score(y_true = test1$my_label, y_pred = predictions_rf, positive = NULL)
F1_RF1





