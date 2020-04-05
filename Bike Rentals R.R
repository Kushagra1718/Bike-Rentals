rm(list = ls())
setwd("C:/Users/HP/Desktop/project2")

#Loading required libraries
#Help for exploring missing data dependencies with minimal deviation.
library(naniar)
#Calculates correlation of variables and displays the results graphically.
library(corrgram)
#Offers a powerful graphics language for creating elegant and complex plots.
library(ggplot2)
#Contains functions to streamline the model training process for complex regressions.
library(caret)
#This package includes functions and data accompanying the book "Data Mining with R.
library(DMwR)
#Recursive partitioning for classification, regression and survival trees.
library(rpart)
library("rpart.plot")
#For random forest
library(randomForest)
#For impact of uncertainities
library(usdm)
#For easily combining clean datasets
library(DataCombine)
#For DecisionTree
library(inTrees)
#collection of R packages designed for data science
library('tidyverse')
#For better visualizations
library('hrbrthemes')
#For better visualizations
library('viridis')


#################################################################################################
#                               Loading Data 
training_data = read.csv('day.csv',header = T,na.strings = c(""," ","NA"))
bckup = training_data    
#################################################################################################

#                                   Exploratory Data Analysis

################################################################################################

#Getting the view and dimension of data
head(training_data,5)
dim(training_data)

# Structure of our dataset
str(training_data)  

# Summary of the data
summary(training_data)

# Extracting datetime
training_data$dteday <- format(as.Date(training_data$dteday,format="%Y-%m-%d"), "%d")

# Removing instant as it's just an indexing id
training_data$instant <- NULL

# Frequency of each unique value
apply(training_data, 2,function(x) length(table(x)))

# Distribution of cnt variable that is also our target variable
hist((training_data$cnt))

# Releant type conversion
cat_var = c('dteday','season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday','weathersit')
num_var = c('temp', 'atemp', 'hum', 'windspeed','casual','registered','cnt')

# Data type conversion function 
typ_conv = function(df,var,type){
  df[var] = lapply(df[var], type)
  return(df)
}
training_data = typ_conv(training_data,cat_var, factor)
############################################################################################

#                                Checking for missing values

############################################################################################

apply(training_data, 2, function(x) {sum(is.na(x))}) 
# Data has no missing values

############################################################################################

#                                 Data Visualization

############################################################################################

box_plot = function(numerical_variables, categorical_variables, dataframe=training_data){
  dataframe %>% #Chaining
    ggplot(aes_string(x = categorical_variables, y = numerical_variables, fill = categorical_variables)) +
    geom_boxplot() +
    geom_jitter(color='black', size=0.4, alpha=0.9) +
    theme_ipsum() +
    theme(
      legend.position = 'top',
      plot.title = element_text(size = 9)
    ) +
    ggtitle(paste("BoxPlot with", categorical_variables, " & ",numerical_variables))
}

box_plot('temp','season',training_data)
box_plot('hum','season')
box_plot('windspeed','weathersit')

box_hist_plot = function(numerical_variables, dataframe=training_data){
  numerical_col = dataframe[,numerical_variables]
#For splitting the screen  
  layout(mat = matrix(c(1,2),nrow = 2, ncol = 1, byrow = TRUE), heights = c(1,8))
#Boxplot formation
  par(mar=c(0, 3.1, 1.1, 2.1)) #Margins
  boxplot(numerical_col , horizontal=TRUE , ylim=c(min(numerical_col),max(numerical_col)), xaxt="n" , col=rgb(0.8,0.8,0,0.5) , frame=F)
  par(mar=c(4, 3.1, 1.1, 2.1))
  hist(numerical_col , breaks=40 , col=rgb(0.2,0.8,0.5,0.5) , border=F , main="" , xlab=paste("Variable value : ",numerical_variables), xlim=c(min(numerical_col),max(numerical_col)))
}

box_hist_plot('temp')
box_hist_plot('atemp')
box_hist_plot('hum')
box_hist_plot('windspeed')
box_hist_plot('casual')
box_hist_plot('registered')
box_hist_plot('cnt')

bar_plot = function(x_col, y_col, fill_col){
    training_data %>%
    ggplot(aes_string(x = x_col, y = y_col, fill = fill_col))+
    geom_bar(position='stack', stat = 'identity')+
    scale_fill_viridis(discrete = T)+
    ggtitle(paste("Bar Plot of",x_col,"on X-Axis,",y_col,"on Y-Axis &", fill_col,"stacked bars."))+
    theme_dark()+
    xlab("")
}


bar_plot('season','hum','weathersit')
bar_plot('mnth','windspeed','weathersit')
bar_plot('season','cnt','weathersit')
bar_plot('weathersit','temp','season')
bar_plot('season','temp','weathersit')


########################################################################################

#                                  Outlier Analysis

########################################################################################

# Imputing outliers 
for(i in c('temp', 'atemp', 'hum', 'windspeed')){
  print(i)
  outv = training_data[,i][training_data[,i] %in% boxplot.stats(training_data[,i])$out]
  print(length(outv))
  training_data[,i][training_data[,i] %in% outv] = NA
}

sum(is.na(training_data))
training_data$hum[is.na(training_data$hum)] = mean(training_data$hum,na.rm = T)
training_data$windspeed[is.na(training_data$windspeed)] = mean(training_data$windspeed, na.rm = T)
# Cross Verifying
sum(is.na(training_data))


##########################################################################################

#                                      Feacture Selection

##########################################################################################
num_var = c('temp', 'atemp', 'hum', 'windspeed','casual','registered','cnt')
corrgram(training_data[,num_var],
         order = F,  
         #As we don't want to reorder
         upper.panel=panel.pie,
         lower.panel=panel.shade,
         text.panel=panel.txt,
         main = 'CORRELATION PLOT')

#Positive correlations are displayed in blue and negative correlations in red color.
#Color intensity and the size of the circle are proportional to the correlation coefficients.

#training_data = subset(training_data, select=-c(atemp,casual,registered))

#Chi-Square test
chi_cat_var = c('dteday','season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday','weathersit')
chi_cat_df = training_data[,cat_var]

for (i in chi_cat_var){
  for (j in chi_cat_var){
    print(i)
    print(j)
    print(chisq.test(table(chi_cat_df[,i], chi_cat_df[,j]))$p.value)
  }
}


########################################################################################

#                                    check multicollearity

########################################################################################
vif(training_data)
training_data = subset(training_data, select=-c(atemp,casual,registered))
training_data = subset(training_data, select=-c(holiday, workingday,dteday))
#Getting final columns
colnames(training_data)
########################################################################################

#                                      Sampling of Data

########################################################################################

#To produce the same result for different instances.
set.seed(17)
t_index = sample(1:nrow(training_data), 0.8*nrow(training_data))
train = training_data[t_index,] 
test = training_data[-t_index,]

# MAPE
mape = function(actual, predict){
mean(abs((actual-predict)/actual))*100
}

########################################################################################

#                                      Linear Regression

########################################################################################

dummy = dummyVars(~., training_data)
dummy_df = data.frame(predict(dummy, training_data))

set.seed(100)
dum_index = sample(1:nrow(dummy_df), 0.8*nrow(dummy_df))
dum_train_df = dummy_df[dum_index,]
dum_test_df = dummy_df[-dum_index,]
lr_model = lm(cnt ~. , data = dum_train_df)
summary(lr_model)
# Forecasting
LR_predict_train = predict(lr_model, dum_train_df[,-32])
plot(dum_train_df$cnt, LR_predict_train,
xlab = 'Actual values',
ylab = 'Predicted values',
main = 'Linear Regression Model')

# Evaluation
postResample(LR_predict_train, dum_train_df$cnt)
mape(dum_train_df$cnt, LR_predict_train)


# Forecasting for test
LR_predict_test = predict(lr_model, dum_test_df[,-32])
plot(dum_test_df$cnt, LR_predict_test,
     xlab = 'Actual values',
     ylab = 'Predicted values',
     main = 'Linear Regression Model')

# Evaluation
postResample(LR_predict_test, dum_test_df$cnt)
mape(dum_test_df$cnt, LR_predict_test)


#########################################################################################

#                                       Decision Tree

#########################################################################################


set.seed(101)
# Model Development
dt_model = rpart(cnt~. , data = train, method = "anova")
summary(dt_model)
plt = rpart.plot(dt_model, type = 5, digits = 2, fallen.leaves = TRUE)

# Forecasting on Train data
DT_Predict_train = predict(dt_model, train[,-9])
plot(train$cnt, DT_Predict_train,
     xlab = 'Actual values',
     ylab = 'Predicted values',
     main = 'Decision Tree Model')

# Evaluation
postResample(DT_Predict_train, train$cnt)
mape(train$cnt, DT_Predict_train)

# Forecasing on Test data 
DT_Predict_test = predict(dt_model, test[,-9])
plot(test$cnt, DT_Predict_test,
     xlab = 'Actual values',
     ylab = 'Predicted values',
     main = 'Decision Tree Model')

# Evaluation
postResample(DT_Predict_test, test$cnt)
mape(test$cnt, DT_Predict_test)

#########################################################################################

#                                       Random Forest

#########################################################################################


set.seed(102)
rf_model = randomForest(cnt ~. , train, importance = TRUE, ntree = 500)
rf_model

# Plotting of error
plot(rf_model)

# Importance of variables
varImpPlot(rf_model)

#Plotting using Random Forest model
RF_predict_train = predict(rf_model, train[,-9])
plot(train$cnt, RF_predict_train,
     xlab = 'Actual values',
     ylab = 'Predicted values',
     main = 'Random Forest model')

# Train result
postResample(RF_predict_train, train$cnt)
mape(train$cnt, RF_predict_train)


#Plotting predict test data using RF model
RF_predict_test = predict(rf_model, test[,-9])
plot(test$cnt, RF_predict_test,
     xlab = 'Actual values',
     ylab = 'Predicted values',
     main = 'Random Forest model')

#Test Result
postResample(RF_predict_test, test$cnt)
mape(test$cnt, RF_predict_test)


