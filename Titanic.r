install.packages(c("e1071","caret","doSNOW","ipred","xgboost"))
install.packages("caret")
library(doSNOW)
library(caret)
library(Amelia)
library(xgboost)
train<-read.csv("train.csv",stringsAsFactors = FALSE)
str(train)
#replace missing value with mode
table(train$Embarked)
any(is.na(train$Embarked))
missmap(train)
train$Embarked[train$Embarked==""]<-"S"
table(train$Embarked)
any(is.na(train$Embarked))

#add afeature for tracking ages

summary(train$Age)
train$MissingAge<-ifelse(is.na(train$Age),"Y","N")

train$FamilySize<-1+train$SibSp+train$Parch
#setup Factor

str(train)

train$Survived<-as.factor(train$Survived)
train$Pclass<-as.factor(train$Pclass)
train$Sex<-as.factor(train$Sex)
train$MissingAge<-as.factor(train$MissingAge)
train$Embarked<-as.factor(train$Embarked)


#setup data to features we wish to keep  it

features<-c("Survived","Pclass","Sex","Age","SibSp",
            "Parch","Fare","Embarked","MissingAge",
            "FamilySize")

train<-train[,features]

str(train)

# create dummy variables for factor 

dummy.Vars<-dummyVars(~.,data=train[,-1])
train.dummy<-predict(dummy.Vars,train[,-1])
View(train.dummy)

#now Impute

pre.Process<-preProcess(train.dummy,method = "bagImpute")

imputed.data<-predict(pre.Process,train.dummy)

View(imputed.data)

train$Age<-imputed.data[,6]
View(train$Age)


#############splitdata
set.seed(1234)

indexes<-createDataPartition(train$Survived,times=1,p=0.7,list = F)
Titanic.train<-train[indexes,]
Titanic.test<-train[-indexes,]

########3examine the proportions of the survived class lable across the dataset
prop.table(table(train$Survived))
prop.table(table(Titanic.train$Survived))
prop.table(table(Titanic.test$Survived))


## setup to caret to performe 10 fold cross validation repeated 3 times
# and to use grid search for optimal model hyparameter values

train.control <- trainControl(method = "repeatedcv",
                              number = 10,
                              repeats = 3,
                              search = "grid")


# Leverage a grid search of hyperparameters for xgboost. See 
# the following presentation for more information:
# 
tune.grid <- expand.grid(eta = c(0.05, 0.075, 0.1),
                         nrounds = c(50, 75, 100),
                         max_depth = 6:8,
                         min_child_weight = c(2.0, 2.25, 2.5),
                         colsample_bytree = c(0.3, 0.4, 0.5),
                         gamma = 0,
                         subsample = 1)
View(tune.grid)

cl <- makeCluster(10, type = "SOCK")

# Register cluster so that caret will know to train in parallel.
registerDoSNOW(cl)

# Train the xgboost model using 10-fold CV repeated 3 times 
# and a hyperparameter grid search to train the optimal model.
caret.cv <- train(Survived ~ ., 
                  data = Titanic.train,
                  method = "xgbTree",
                  tuneGrid = tune.grid,
                  trControl = train.control)
stopCluster(cl)

#examine carets preprosseing result
caret.cv

preds<-predict(caret.cv,Titanic.test)

confusionMatrix(preds,Titanic.test$Survived)



