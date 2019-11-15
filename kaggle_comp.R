install.packages("tidyverse")
install.packages("corrplot")
install.packages("caret")
install.packages("dplR")
install.packages("caTools")
install.packages("broom")
library(tidyverse)
library(corrplot)
library(caret)
library(dplR)
library(caTools)
library(broom)
library(randomForest)
library(rpart)
library(modelr)

#loading data

df <- read.csv("vw_kaggle_train.csv", stringsAsFactors = F)
#df$my_label <- as.factor(df$my_label) - can do this later if needed 
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

feature_correlations1 <- cor(df[1:28])
print(feature_correlations1)
corrplot(feature_correlations1,method="circle")
feature_highcorrelation <- findCorrelation(feature_correlations1, cutoff=0.75) #find highly correlated features, generally should remove reatures with greater than 75% correlation
print(feature_highcorrelation)
colnames(df[c(7,15,13,14,11,16,10,12,6,26,28,24,23,22,25,18)]) #these indexes represent variables with absolutle correlation above 75%. Can remove from training data

df_clean <- select(df,-c(7,15,13,14,11,16,10,12,6,26,28,24,23,22,25,18))
dim(df_clean)
head(df_clean)

#holding back 25% data for testing
#split train and test data 
set.seed(123) 
sample = sample.split(df_clean, SplitRatio = .75)
train1 = subset(df_clean, sample == TRUE)
test1  = subset(df_clean, sample == FALSE)
dim(train1)
dim(test1)

str(train1$my_label)
train1$my_label <- as.factor(train1$my_label)
str(test1$my_label)
test1$my_label <- as.factor(test1$my_label)

#training DT
set.seed(123)
dt <- rpart(my_label ~ activePower+activePowerDelta+reactivePower+voltage+
                         phase+transient8+transient10+harmonicDelta1+harmonicDelta2+
                         harmonicDelta8, data=train1, method = "class")
print(dt)

predictions_dt <- predict(dt, test1, type = "class")
confusionMatrix(predictions_dt, test1$my_label)

#training a random forest 

forest1 <- randomForest(my_label ~ activePower+activePowerDelta+reactivePower+voltage+
                          phase+transient8+transient10+harmonicDelta1+harmonicDelta2+
                          harmonicDelta8, data=train1, importance = TRUE)










