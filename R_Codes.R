# Load the data

park <- read.csv("parkinsons.data",sep = ",", header = T)

str(park)


set.seed(123)

library(caret)
library(MASS)
library(car)


#### Finding correlated variables

corr <- cor(park[-c(1,18)])
corrDF <- expand.grid(row = names(park)[-c(1,18)], col = names(park)[-c(1,18)])
corrDF$correlation <- as.vector(corr)
levelplot(correlation ~ row+ col, corrDF,scales=list(x=list(cex=.4),y=list(cex=.5)))

#convert response variable to factor

park$status <- as.factor(park$status)

# Splitting the data into train and test. The distribution of the response variable is maintained in both the test and train data sets


train.prop=0.80
trainIndex <- createDataPartition(park$status, p=train.prop, list=FALSE)
data_train <- park[ trainIndex,]
data_test <- park[-trainIndex,]

## Logistic Regression with all the features available in the data

fullmod = glm(status ~ .,data = data_train[-1],family=binomial)

# StepWise Regression to remove non useful features. We do backwards regression here.

backwards = stepAIC(fullmod,trace=0)

# Predicting on test data
prd.test <- predict(backwards,newdata = data_test,type = "response")
prd.train <- predict(backwards,newdata = data_train,type = "response")

conf_matrix_test <- table(data_test$status,ifelse(prd.test > 0.5 ,1,0))
conf_matrix_train <- table(data_train$status,ifelse(prd.train > 0.5 ,1,0))


# Area under the curve

library("ROCR")
pred <- prediction(prd.test, data_test$status)
perf <- performance(pred,"tpr","fpr")
plot(perf,col="black",lty=3, lwd=3)
auc <- performance(pred,"auc")
auc

library("ROCR")
pred <- prediction(prd.train, data_train$status)
perf <- performance(pred,"tpr","fpr")
plot(perf,col="black",lty=3, lwd=3)
auc <- performance(pred,"auc")
auc


# Building RPART models

rpart.fit <- rpart(status ~ ., data=data_train[-1], method="class",parms=list(split="information", loss=matrix(c(0,1,3,0), byrow=TRUE, nrow=2)))

# Choose the best cp which has low cross validation error and prune the tree
printcp(rpart.fit)
plotcp(rpart.fit)

rpart.prune = prune(rpart.fit, cp = 0.059829)

# Plotting the final tree
plot(rpart.prune, uniform = TRUE)
text(rpart.prune, use.n = TRUE, cex = 0.75)

#Performance
pred.rpart.test <- predict(rpart.prune,newdata = data_test[-1],type = "class")
conf_matrix_test_rpart <- table(data_test$status,pred.rpart.test)


pred.rpart.train <- predict(rpart.prune,newdata = data_train[-1],type = "class")
conf_matrix_train_rpart <- table(data_train$status,pred.rpart.train)

#### Building Random Forest Models ####

# Grid Search for tuninng the mtry variable

control <- trainControl(method="repeatedcv", number=10, repeats=3, summaryFunction = twoClassSummary,classProbs = TRUE)
set.seed(1234)
tunegrid <- expand.grid(.mtry=c(1:10))
rf_gridsearch <- train(status1~., data=data_train[-c(1,18)], method="rf", metric="Sens", tuneGrid=tunegrid, trControl=control)
print(rf_gridsearch)
plot(rf_gridsearch)

pred.rf.test <- predict(rf_gridsearch,newdata = data_test[-c(1,18)],type = "raw")
conf_matrix_test_rf <- table(data_test$status1,pred.rf.test)

pred.rf.train <- predict(rf_gridsearch,newdata = data_train[-c(1,18)],type = "raw")
conf_matrix_train_rf <- table(data_train$status1,pred.rf.train)

