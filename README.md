# KingCounty_PredicitveModel
library(AppliedPredictiveModeling)
library(caret)
## load data
KC_data <-read.csv("/Users/mubashirsultan/Documents/R/Week12/kc_house_data.csv")


#Removed homes with prices above $1.5M
KC_data <- subset(KC_data, price <= 1500000) 

#Classify the dataset by adding a new factor column: Category
KC_data$Category <- factor(ifelse(KC_data$price < 500000, "low", ifelse(KC_data$price < 1000000, "medium","high")))

#View Data
str(KC_data)

# Get all column names from the dataset: all_cols
all_cols <- names(KC_data)

#Create a dataframe for category column
Category <- data.frame(KC_data[,22])

#Remove first 3 and last columns (ID, Date, and Category)
KC_data <-KC_data[,-c(1:3,22)]

# Identify near zero variance predictors: remove_cols
remove_cols <- nearZeroVar(KC_data)

# Remove predictors that have low variance
data <- KC_data[,-remove_cols]

# Remove predictors that are highly correlated with other predictors
Cor1 <- cor(KC_data, use="pairwise.complete.obs")
summary(Cor1[upper.tri(Cor1)])
data2 <- findCorrelation(Cor1, cutoff = 0.75, exact = TRUE)
data <- data[,-data2]
Cor2 <- cor(data)
summary(Cor2[upper.tri(Cor2)])

preProcessOut <- preProcess(data, method=c('center', 'scale', 'YeoJohnson'))
str(preProcessOut)
KC_data2 <- predict(preProcessOut, data)

# Transforming to resolve Skewness
library(e1071)
apply(KC_data2,2,skewness)

#add Category back into the KC_Data data frame
KC_data2$Category <- Category[,1]

str(KC_data2)

# Repeated CV
set.seed(1324)


library(snow)
library(doSNOW)

cl <- makeCluster(3, type = "SOCK")
registerDoSNOW(cl)

grep('Category', names(KC_data2))

ctrl <- rfeControl(functions = ldaFuncs,
                   method = "repeatedcv",
                   n = 10,
                   repeats = 10,
                   verbose = FALSE)

subsets <- 1:(ncol(KC_data2)-14)

#Linear Discriminant Analysis
set.seed(1324)

rfeOut <- rfe(KC_data2[,-14], KC_data2[,14],
              sizes = subsets,
              rfeControl = ctrl)

stopCluster(cl)

rfeOut
plot(rfeOut)
plot(KC_data2[,c('grade', 'sqft_living15')], col=KC_data2[,14])


## Confusion Matrix 

predictedCategory <- predict(rfeOut, KC_data2)

confusionMatrix(data = predictedCategory$pred, 
                reference = KC_data2$Category)

#Report the performance of each model on resampling cross-validation
postResample(predictedCategory, KC_data2$Category)


##  Neural Network Model

cl <- makeCluster(3, type = "SOCK")
registerDoSNOW(cl)

grep('Categort', names(KC_data2))

set.seed(1324)
fitctrl <- trainControl(method = "repeatedcv",
                        n = 10,
                        repeats = 10)

set.seed(1324)
nnetOut <- train(KC_data2[,-14], KC_data2[,14], method='nnet',
                 trControl = fitctrl,
                 tuneGrid = expand.grid(size=1:10, 
                                        decay=c(0,.1, 1, 2)))

stopCluster(cl)

nnetOut
plot(nnetOut)

## Confusion Matrix

predictedOil2 <- predict(nnetOut, KC_data2)

confusionMatrix(data = predictedOil2, 
                reference = KC_data2$Category)


## Lift Curve

nnetpred <- predict(nnetOut, KC_data2, type='prob')

postResample(predictedOil2,KC_data2$Category)
