#Airbnb - New User Bookings
#Author: Tyrone Cragg
#Date: April 2016

#Load required libraries
library(data.table)
library()
library(tidyr)
library(bit64)
library(Matrix)
library(xgboost)

#Load training and testing data
train = fread("input/train_users_2.csv")
test = fread("input/test_users.csv")

#Load sessions data
sessions = fread("input/sessions.csv")

#Remove session rows with no user_id
sessions = sessions[user_id != ""]

#Count number of each action by user_id
user_actions = sessions[, list(count=.N), by=c("user_id", "action")]
user_actions = spread(data=user_actions, key=action, value=count, fill=0, sep="_")

#Count number of each action_detail by user_id
user_action_details = sessions[, list(count=.N), by=c("user_id", "action_detail")]
user_action_details = spread(data=user_action_details, key=action_detail, value=count, fill=0, sep="_")

#Remove unused data to clear memory
rm(sessions)
gc()

#Convert timestamp_first_active into date format
train[, date_first_active := paste(substr(timestamp_first_active, 0, 4), substr(timestamp_first_active, 5, 6),
                          substr(timestamp_first_active, 7, 8), sep="-")]
test[, date_first_active := paste(substr(timestamp_first_active, 0, 4), substr(timestamp_first_active, 5, 6),
                          substr(timestamp_first_active, 7, 8), sep="-")]

#Convert to date
train[, date_first_active := as.Date(date_first_active)]
test[, date_first_active := as.Date(date_first_active)]

#No session data before 2014-01-01 - remove this training data
train = train[date_first_active >= 20140101000000]

#Add country_destination in order to combine training and testing data
test[, country_destination := "Test"]

#Combine training and testing data
data = rbind(train, test)

#Remove unused data to clear memory
rm(train, test)
gc()

#Set keys for joining
setkey(data, id)
setkey(user_actions, user_id)
setkey(user_action_details, user_id)

#Join data
data = user_actions[data]
data = user_action_details[data]

#Remove unused data to clear memory
rm(user_actions, user_action_details)

#Set NAs in merged data to missing value for xgboost (no session counts)
data[is.na(data)] = -999

#Observe age distribution
hist(data$age)
summary(data$age)

#Clean age
data[, age := ifelse(age < 15, -1, age)]
data[, age := ifelse(age > 104, -1, age)]

#Convert data types
data[, date_account_created := as.Date(date_account_created)]
data[, month_account_created := as.integer(format(date_account_created, "%m"))]
data[, month_first_active := as.integer(format(date_first_active, "%m"))]

#Get training and testing indices
train_indices = data[, .I[country_destination != "Test"]]
test_indices = data[, .I[country_destination == "Test"]]

#Add variable for days available for user to book based on maximum date in the data set
data[train_indices, DaysAvailable := as.numeric(as.Date("2014-06-30") - date_account_created)]
data[test_indices, DaysAvailable := as.numeric(as.Date("2014-09-30") - date_account_created)]

#Extract target variable
train_country_destination = data[train_indices, country_destination]

#Remove columns
data[, c("date_account_created", "date_first_active", "timestamp_first_active", "date_first_booking",
         "country_destination") := NULL]

#Set NAs to missing value
data[is.na(data)] = -999

#Sparsify data
data = sparse.model.matrix(~. -1, data)

#Split training and testing data
train = data[train_indices]
test = data[test_indices]

#Remove unused data to clear memory
rm(data)
gc()

#Recode target to numeric for xgboost
library(car)
train_country_destination = recode(train_country_destination,
                                   "'NDF'=0; 'US'=1; 'other'=2; 'FR'=3; 'CA'=4; 'GB'=5; 'ES'=6;
                                   'IT'=7; 'PT'=8; 'NL'=9; 'DE'=10; 'AU'=11;")

#Extract names of training variables
train_names = names(train)

#Define evaluation metric
NDCG5 = function(preds, dtrain) {
  labels = getinfo(dtrain,"label")
  num.class = 12
  pred = matrix(preds, nrow = num.class)
  top = t(apply(pred, 2, function(y) order(y)[num.class:(num.class-4)]-1))
  
  x = ifelse(top == labels,1,0)
  dcg = function(y) sum((2^y - 1)/log(2:(length(y)+1), base = 2))
  ndcg = mean(apply(x,1,dcg))
  return(list(metric = "ndcg5", value = ndcg))
}

#Set best parameters
param = list(objective = "multi:softprob",
             num_class = 12,
             booster = "gbtree",
             eta = 0.2,
             max_depth = 6,
             subsample = 0.85,
             colsample_bytree = 0.66
)

#Create xgb.DMatrix for xgboost
dtrain = xgb.DMatrix(data=data.matrix(train), label=train_country_destination, missing=-1)

#Set early.stop.round for xgboost
early.stop.round = 30

#Run 5-fold cross validation xgboost
set.seed(8)
XGBcv = xgb.cv(params = param,
                data = dtrain, 
                nrounds = 200,
                verbose = 1,
                early.stop.round = early.stop.round,
                nfold = 5,
                feval = NDCG5,
                maximize = T,
                prediction = T
)

# PredTrain = XGBcv$pred
# write.csv(PredTrain, "XGB3PredTrain1.csv", quote=F, row.names=F)

#Extract nrounds of best iteration
nrounds = length(XGBcv$dt$test.ndcg5.mean) - early.stop.round

#Run xgboost on full data set
XGB = xgboost(params = param,
                data = dtrain,
                nrounds = nrounds,
                verbose = 1,
                eval_metric = NDCG5,
                maximize = T
)

# Compute and plot feature importance
Importance = xgb.importance(trainNames, model = XGB)
xgb.plot.importance(Importance[1:10,])

#Predict on test set
PredTest = predict(XGB, data.matrix(test[,trainNames]), missing=-1)

#Reshape predictions
Predictions = as.data.frame(matrix(PredTest, nrow=12))

#Recode to original destination names
rownames(Predictions) = c('NDF','US','other','FR','CA','GB','ES','IT','PT','NL','DE','AU')

#Extract top 5 predictions for each testing row
Predictions_Top5 = as.vector(apply(Predictions, 2, function(x) names(sort(x)[12:8])))

#Create test set ID vector
testId = testMerged$id
testIdMatrix = matrix(testId, 1)[rep(1,5), ]
testIds = c(testIdMatrix)

#Create submission
Submission = NULL
Submission$id = testIds
Submission$country = Predictions_Top5

#Save submission
Submission = as.data.frame(Submission)
write.csv(Submission, "XGB.csv", quote=F, row.names=F)
