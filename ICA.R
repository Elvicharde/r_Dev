#  Title: ICA Machine Learning Task
# Author: OGUNTUASE VICTOR
#   Date: 26/04/2021" 
# e-Mail: freelanceel0@gmail.com

################################################################################
#                        1.0 INTRODUCTION SECTION
#_______________________________________________________________________________
# 1.1. DATASET DESCRIPTION:

# The Telecoms Dataset has been selected for this assessment task based on the
# following criteria:
#
# (1) The Dataset is publicly available/accessible via kaggle.com through this 
#     link https://www.kaggle.com/radmirzosimov/telecom-users-dataset. 
#     Also, it contains real-world observations.
#
# (2) The Dataset exceeds the minimum requirements: at least 1000 observations 
#     and 5 attributes/features, as described for the assessment.
#_______________________________________________________________________________

# 1.2. PROBLEM DEFINITION

# This Dataset contains observations about customers of a mobile network service 
# provider who intends to reduce customer dissatisfaction/disengagement. A  
# variable "churn" has been used to describe the rate of service disengagement  
# of customers, and it is a feature/variable in the Telecoms-Dataset. 
# 
# The service provider, in the hope of reducing this "churn" across its 
# subscribers, requires relationships to be observed across all the attributes
# of the resulting "Telecoms-Dataset". This is to discover opportunities for 
# achieving an overall reduced "churn"/customer disengagement.
#_______________________________________________________________________________

# 1.3 APPLICABLE MACHINE LEARNING ALGORITHMS

# The following machine learning algorithms are applicable to solving the  
# defined problem(s): Logistic regression, Decision tree classification, Random 
# Forest classification, K-Nearest Neighbour, and Naive Bayes classification  
# algorithms.
# The choice of algorithms is based on the observation of the data during  
# analysis, which informs that the intended prediction is of a classification  
# type (the output/dependent variable "churn" is discrete/categorical).
# 
# However, Logistic regression, Decision tree classification, and Random forest 
# classification methods were used in solving this problem.
#_______________________________________________________________________________
################################################################################
#                             2.0 DATA PRE-PROCESSING, 
#                       ANALYSIS, AND VISUALIZATION SECTION
#_______________________________________________________________________________
# 2.1. DATA EXPLORATION

# Preliminaries
# Clearing the console and workspace

cat("\f")      # clears console
rm(list=ls())  # clears the workspace

# Importing relevant packages/libraries: ggplot2, dplyr, caTools, Caret, rpart,
# rpart.plot, randomForest, gridExtra

packages <- c("ggplot2","dplyr","caTools",
              "gridExtra","rpart","rpart.plot","caret",
              "randomForest")

# These packages are useful in the following ways: 

# ggplot2 is excellent for creating visualizations in R i.e. plots, charts, etc.

# dplyr is for data wrangling/manipulation i.e. data cleansing/adjustment.

# caTools is a library that offers useful tools like the sample.split() function
# which makes creating training and testing data simpler.

# Caret offers the use of the "train" and "traincontrol()" functions for working
# with decision trees. Also, it's confusionMatrix() function offers a convenient
# approach to drawing up confusion matrices with useful statistics/calculations.

# rpart and rpart.plot offer decision tree training and visualization.
# It allows viewing the various splits and decisions taken during the 
# train/test process. 

# gridExtra allows for customizing ggplots in grid format.

# randomForest is used for implementing the random forest classification.


# Dynamically loading the needed packages using the require() function.

lapply(packages, require, character.only = TRUE)    
                                                   

# First, the Dataset is imported from the working directory for initial 
# observation.

## IMPORTING THE TELECOMS DATASET FOR EXPLORATORY DATA ANALYSIS (EDA)

data_set <- read.csv("telecom_users.csv") # relative path to working directory 
                                          # is implied for the import

## EXPLORING THE DATA

head(data_set[,1:11])               # Checking the first 6 observations of the 
                                    # first 11 features; truncated for display 
                                    # aesthetics


# Now, getting important numerical statistics about the data. This will inform 
# about the present features/attributes and their usefulness or otherwise.

str(data_set)          # Observing data structure, trends, and details.


# It can be observed that the data has 5986 observations/attributes and 22 
# variables. Also, all the variables are of two data-types: integers and 
# characters.
# Most of the attributes in this Dataset are "Categorical", with only 
# "tenure, Monthly charges, and Total charges" being "Continuous".

# GETTING FURTHER INFORMATION ON THE DATA

summary(data_set)      # Getting numerical summaries of the data.

#_______________________________________________________________________________

# 2.2. CLEANING THE IMPORTED DATA

# As is usually the case, not all the variables in this Dataset are relevant.
# For example, the "x" and "customerID" variables are not of any significance to
# solving the task at hand and can be removed using dplyr's select.

data_set <- select(data_set, -(c("X","customerID")))
head(data_set)

# Now we have 20 features/attributes.
## Further observation of the Dataset informs that the categorical data
#  takes one of two or three possible values. However, all the three-valued
#  categorical data can be reduced to two i.e. "OnlineSecurity, 
#  MultipleLines, Online Backup, etc".

fix_data <- function(column){
  ## This function accepts a column with categorical data and converts them
  ## to "Yes" or "No" based on a condition.
  ifelse(column %in% c("Yes","Fiber optic","DSL","1"), column <-"Yes",
         column <-"No") 
}   

affected_columns <- c("SeniorCitizen", "Partner", "Dependents", "PhoneService",
                      "MultipleLines", "InternetService", "OnlineSecurity",
                      "OnlineBackup", "DeviceProtection", "TechSupport",
                      "StreamingTV", "StreamingMovies", "PaperlessBilling",
                      "Churn")

cleaned <- sapply(data_set[affected_columns],  #using sapply() to auto-iterate
                  fix_data)                   
head(cleaned, 10)    # Viewing the changed data before committing to data_set

# Since we will use our Dataset for machine learning puporses, it is essential  
# to convert the categorical variables to factors.

cleaned <- data.frame(cleaned)
cleaned[affected_columns] <- lapply(cleaned[affected_columns],factor)
str(cleaned)
summary(cleaned)
data_set[affected_columns] <- cleaned    #Merging with full .

## CHECKING FOR NAs
# We still need to confirm there are no missing values or NAs in the Dataset

columns_with_NAs <- colnames(data_set)[apply(data_set,2,anyNA)]
print(paste("The following columns have NAs/missing data: ",columns_with_NAs))

# Since the NA only occured in the TotalCharges column, we can check how 
# many NAs occurred, and the rows of occurence to get more information.

sum(is.na(data_set$TotalCharges))          # Count NAs in $TotalCharges

data_set[is.na(data_set$TotalCharges),]    #Getting the rows with NAs

# The NAs occured in rows: `357,635,2772,3087,3256,4327,5376,5383,5696,5952`. 
# It can also be observed that the customers with NAs still have an active 
# subscription, with most of them running two-year contracts and no *CHURN*

# Since we have the NA in a column with numeric data, we could replace the 
# missing values with the mean/median of that column. This time, we use the 
# mean.

mean_val <- mean(data_set$TotalCharges, na.rm = TRUE) # mean of TotalCharges 

# Implementing the change/replacements
cleaned <- ifelse(is.na(data_set$TotalCharges),mean_val,data_set$TotalCharges)
data_set$TotalCharges <- cleaned   # Merging with the Dataset

# Now that the data is fully cleaned, we can go ahead and do some analysis 
# via visualizations with ggplot2

#_______________________________________________________________________________

# 2.3. DATA ANALYSIS

# The initial insight offered by the function(s) `*str()*`and `*summary()*, 

# While interesting, is not adequate to understand the data.
# With the clean Dataset, we can visualize some trends via barplots, histograms, 
# and categorical plots.
# The plots are created using the ggplot2 library.

# Observing the target feature, CHURN

ggplot(data_set, aes(x = Churn)) +
  geom_bar() +
  geom_text(aes(y = ..count.. -300, 
                label = paste0(round(prop.table(..count..),4) * 100, '%')), 
            stat = 'count', 
            position = position_dodge(.1), 
            size = 4)+
  labs(y = "Observations", title = "                           CHURN ACROSS ALL CUSTOMERS")


# It can be observed that 73.49% of the customers had no churn. However, the 
# remaining 26.51% will still be a significant concern for the service provider.
# We can also observe the other categorical data for similar overview.

# Grid plot of Gender, Senior citizen, Partner, and dependents columns

#Gender plot
plot1 <- ggplot(data_set, aes(x = gender)) +
  geom_bar() +
  geom_text(aes(y = ..count.. -300, 
                label = paste0(round(prop.table(..count..),4) * 100, '%')), 
            stat = 'count', 
            position = position_dodge(.1), 
            size = 4) + labs(y = "Observations")


#Senior citizen plot
plot2 <- ggplot(data_set, aes(x = SeniorCitizen)) +
  geom_bar() +
  geom_text(aes(y = ..count.. -300, 
                label = paste0(round(prop.table(..count..),4) * 100, '%')), 
            stat = 'count', 
            position = position_dodge(.1), 
            size = 4) + labs(y = "Observations")

#Partner plot
plot3 <- ggplot(data_set, aes(x = Partner)) +
  geom_bar() +
  geom_text(aes(y = ..count.. -300, 
                label = paste0(round(prop.table(..count..),4) * 100, '%')), 
            stat = 'count', 
            position = position_dodge(.1), 
            size = 4) + labs(y = "Observations")

#Dependents plot
plot4 <- ggplot(data_set, aes(x = Dependents)) +
  geom_bar() +
  geom_text(aes(y = ..count.. -300, 
                label = paste0(round(prop.table(..count..),4) * 100, '%')), 
            stat = 'count', 
            position = position_dodge(.1), 
            size = 4) + labs(y = "Observations")

# grid plot
grid.arrange(plot1, plot2, plot3, plot4, nrow=2)


# The resulting plots inform of a certain trend in the data:  
#   
#   1. The gender and partner features are closely distributed between "Yes" 
#      and "No". This may imply a minimal impact on the overall churn  
# 
#   2. The categorical plot of Senior Citizens and Dependent features show a 
#      large variation between "Yes" and "No". These may be significant factors  
# 
# 
# Further plots on the various services rendered to the customers may offer 
# useful information as well. This time, the plots are further differentiated 
# by the "churn" variable.

# Grid plot of the services offered

# Phone service plot
p5 <- ggplot(data_set, aes(x = PhoneService)) +
  geom_bar() +
  geom_text(aes(y = ..count.. -300, 
                label = paste0(round(prop.table(..count..),4) * 100, '%')), 
            stat = 'count', 
            position = position_dodge(.1), 
            size = 3) + labs(y = "Observations")

# Multiple phone lines 
p6 <- ggplot(data_set, aes(x = MultipleLines)) +
  geom_bar() +
  geom_text(aes(y = ..count.. -300, 
                label = paste0(round(prop.table(..count..),4) * 100, '%')), 
            stat = 'count', 
            position = position_dodge(.1), 
            size = 3) + labs(y = "Observations")

# Internet service 
p7 <- ggplot(data_set, aes(x = InternetService)) +
  geom_bar() +
  geom_text(aes(y = ..count.. -300, 
                label = paste0(round(prop.table(..count..),4) * 100, '%')), 
            stat = 'count', 
            position = position_dodge(.1), 
            size = 3) + labs(y = "Observations")

# Online security 
p8 <- ggplot(data_set, aes(x = OnlineSecurity)) +
  geom_bar() +
  geom_text(aes(y = ..count.. -300, 
                label = paste0(round(prop.table(..count..),4) * 100, '%')), 
            stat = 'count', 
            position = position_dodge(.1), 
            size = 3) + labs(y = "Observations")

# Online backup 
p9 <- ggplot(data_set, aes(x = OnlineBackup)) +
  geom_bar() +
  geom_text(aes(y = ..count.. -300, 
                label = paste0(round(prop.table(..count..),4) * 100, '%')), 
            stat = 'count', 
            position = position_dodge(.1), 
            size = 3) + labs(y = "Observations")

# Device Protection 
p10 <- ggplot(data_set, aes(x = DeviceProtection)) +
  geom_bar() +
  geom_text(aes(y = ..count.. -300, 
                label = paste0(round(prop.table(..count..),4) * 100, '%')), 
            stat = 'count', 
            position = position_dodge(.1), 
            size = 3) + labs(y = "Observations")

# Tech Support 
p11 <- ggplot(data_set, aes(x = TechSupport)) +
  geom_bar() +
  geom_text(aes(y = ..count.. -300, 
                label = paste0(round(prop.table(..count..),4) * 100, '%')), 
            stat = 'count', 
            position = position_dodge(.1), 
            size = 3) + labs(y = "Observations")

# Streaming TV 
p12 <- ggplot(data_set, aes(x = StreamingTV)) +
  geom_bar() +
  geom_text(aes(y = ..count.. -300, 
                label = paste0(round(prop.table(..count..),4) * 100, '%')), 
            stat = 'count', 
            position = position_dodge(.1), 
            size = 3) + labs(y = "Observations")

# Streaming Movies
p13 <- ggplot(data_set, aes(x = StreamingMovies)) +
  geom_bar() +
  geom_text(aes(y = ..count.. -300, 
                label = paste0(round(prop.table(..count..),4) * 100, '%')), 
            stat = 'count', 
            position = position_dodge(.1), 
            size = 3) + labs(y = "Observations")

# grid 1
grid.arrange(grobs = list(p5, p6, p7), nrow = 1)

#grid 2
grid.arrange(grobs = list(p8, p9, p10), nrow = 1)

#grid 3
grid.arrange(grobs = list(p11, p12, p13), nrow = 1)


# The resulting plots show that most of the customers have phone service with 
# single phone lines, internet access/subscription, no online security or 
# backup, no device protection or tech support, and they don't stream TV or 
# movies. There was a significant variation in the plots, with the most 
# variation in the "phone service and Internet service" plots.

# Furthermore, we can visualize the other categories: contract, paperless 
# billing method, and payment method.


# Plotting the remaining categorical data

#Contract status 
p14 <- ggplot(data_set, aes(x = Contract)) +
  geom_bar() +
  geom_text(aes(y = ..count.. -200, 
                label = paste0(round(prop.table(..count..),4) * 100, '%')), 
            stat = 'count', 
            position = position_dodge(.1), 
            size = 3) + labs(y = "Observations")

#Paperless billing 
p15 <- ggplot(data_set, aes(x = PaperlessBilling)) +
  geom_bar() +
  geom_text(aes(y = ..count.. -200, 
                label = paste0(round(prop.table(..count..),4) * 100, '%')), 
            stat = 'count', 
            position = position_dodge(.1), 
            size = 3) + labs(y = "Observations")


#grid 4
grid.arrange(grobs = list(p14, p15), nrow=1)

#Payment method
p16 <- ggplot(data_set, aes(x = PaymentMethod)) +
  geom_bar() +
  geom_text(aes(y = ..count.. -200, 
                label = paste0(round(prop.table(..count..),4) * 100, '%')), 
            stat = 'count', 
            position = position_dodge(.1), 
            size = 3) + labs(y = "Observations")
p16


# The above plots inform of a high variability in the contract feature, for 
# which many subscribers fall into the "month-month" category. This, combined 
# with tenure, may inform of users tendency to churn or otherwise based on their 
# loyalty to the network. Payment method also offers a significant insight, with 
# most subscribers adopting the electronic check option.  
# 
# Finally, we can observe the continuous variables via bar plots and histograms.

# Observing tenure

ggplot(data_set, aes(x = tenure)) + geom_histogram(aes (colour = Churn)
                    ,bins = 30) + labs(y = "Observations")

# It can be observed that customers with the shortest tenure were more likely 
# to churn than those that have used the network for longer. Similarly, we can 
# determine if the "CHURN" is financially motivated by observing the monthly 
# charges and total charges features.

ggplot(data_set, aes(x = MonthlyCharges)) + geom_histogram(aes (colour = Churn),
                    bins = 30) + labs(y = "Observations")

ggplot(data_set, aes(x = TotalCharges)) + geom_histogram(aes (colour = Churn)
                 ,bins = 30) + labs(y = "Observations")


# The MonthlyCharges and TotalCharges plots show some interesting insights into 
# the data. First, due to the distribution of the MonthlyCharges data, it can be 
# deduced that the "CHURN" is not financially motivated. This is because both 
# high and low paying subscribers (monthly) had low CHURN tendencies. However, 
# the TotalCharges plot inform that subscribers that have paid the most i.e 
# $2500+ were less likely to churn. THe high churn rate among subscribers that 
# have paid less that $2500 in total may then be attributed to the effect of 
# "tenure".

# > New subscribers have a higher tendency to churn, given the contribution of 
# other factors.
#_______________________________________________________________________________
################################################################################
#                             3.0 MACHINE LEARNING SECTION 
#_______________________________________________________________________________

# Here, we are going to apply the three choice machine learning algorithms to 
# the cleaned Dataset. To achieve this, the data needs to first be split into 
# training and testing data. One of the already loaded libraries "caTools" 
# offers this flexbility in form of "sample.split".

# SPlitting the Dataset

set.seed(123)   #setting random generator state to allow for reproducibility
data_split <- sample.split(data_set, SplitRatio = 0.7)
train_data <- subset(data_set, data_split == "TRUE")
test_data <- subset(data_set, data_split == "FALSE")


# 3.1. LOGISTIC REGRESSION

# Creating and training a logistic regression model

logR_model <- glm(Churn ~., data = train_data, family = "binomial")
summary(logR_model)    # Viewing model details and features.

# The results of the model training show the most significant attributes of our 
# trained model. As expected, Tenure, Contract-Type, PhoneService = "Yes", 
# OnlineSecurity = "Yes", TechSupport = "Yes", and "MonthlyCharges" were the 
# most significant features.

# Testing the model

prediction <- predict(logR_model, test_data, type = "response")

# Saving the predictions as a factor for comparing with actual values.
# 0.5 was chosen as a threshold because of the binary nature of the exepected
# outcome. Results greater or less than a probability of 0.5 indicate Churn and 
# "no Churn" respectively.

prediction <- factor(ifelse(prediction > 0.5,"Yes", "No")) 

# Drawing up a confusion matrix for evaluation

conf_mat <- confusionMatrix(prediction, test_data$Churn)
conf_mat

# The results shows that the model is better at predicting if a customer will 
# stay with the network than otherwise. However, the model offers good 
# prediction accuracy and can predict correctly about 80% of the time.


# Creating a function for displaying the error rates: overall error, type-1 
# error(False Positives), and type-2(False Negatives) error.

errors <- function (conf_mat){
  # Getting the required values from the overall_values() function defined 
  # below.
  
  #overall error is a residual of overall accuracy
  ov_error <- (round(1 - as.numeric(conf_mat$overall[1]), 4) * 100)
  
  # specificity residual
  t1_error <- (round(1 - as.numeric(conf_mat$byClass[2]), 4) * 100)
  
  #sensitivity residual
  t2_error <- (round(1 - as.numeric(conf_mat$byClass[1]), 4) * 100)
  
  return <- list(ov_error, t1_error, t2_error)
}

all_errors <- unlist(errors(conf_mat))

logisticReg_overall_error <- all_errors[1]
logisticReg_type1_error <- all_errors[2]
logisticReg_type2_error <- all_errors[3]

cat(paste0("\nOverall error: ",logisticReg_overall_error,"%\n Type 1 error: ", 
  logisticReg_type1_error,"%\n Type 2 error: ",logisticReg_type2_error,"%\n"))

#_______________________________________________________________________________

# 3.2. Decision Tree

# Creating and training the model

tree_model <- rpart(Churn ~., data = train_data)  # training the decision tree
summary(tree_model)

# The summary of the decision tree training informs of some agreement with what 
# was observed from the logistic regression analysis. In this case, features 
# are ranked according to their importance with Contract,Tenure, 
# and Total Charges being the most important variables (accounting for 65% 
# of the variations in "Churn").

rpart.plot(tree_model) # Visualizing the decisions
prp(tree_model)    # Another plot showing the choices


# Testing the model

prediction <- predict(tree_model, test_data, type = "class")
summary(prediction)

# Confusion matrix for evaluating the decision tree model

conf_mat <- confusionMatrix(prediction, test_data$Churn)
conf_mat

# Here, we can see that the accuracy of the decision tree model is at 
# approximately 77%. This is also a good accuracy but slightly lower than what 
# the logistic regression offers.

# Printing other results using the "errors()" function, earlier created.

all_errors <- unlist(errors(conf_mat))

dectree_overall_error <- all_errors[1]
dectree_type1_error <- all_errors[2]
dectree_type2_error <- all_errors[3]

cat(paste0("\nOverall error: ",dectree_overall_error,"%\n Type 1 error: ", 
           dectree_type1_error,"%\n Type 2 error: ",dectree_type2_error,"%\n"))

#_______________________________________________________________________________

# 3.2. Random Forest

# This is the final machine learning model for examining this Dataset. This 
# logically follows after the decision trees because it is a method that 
# leverages on the hybridization of multiple decision trees to achieve a more 
# definitive result. 
# 
# > A random forest is an ensemble of decision trees

# Prior to building and deploying a random forest algorithm, two key parameters 
# must be defined:
#   
# 1. The number of variables/choices to use for splitting the decisions at each 
# node in the Random Forest trees: this is controlled via the Mtry argument 
# of the `RandomForest()` function.  
# 
# 2. The number of trees that make-up the Random Forest. This is adjusted with 
# the ntree argument of the `RandomForest()` function.  
# 
# Since the choice of these parameters can impact the accuracy of the Random 
# Forest classifier, we take a systematic approach by running preliminary 
# simulations using tools in the `caret library`.

set.seed(123)
simulation_control <- trainControl(method = "repeatedcv", number = 5, 
                                   repeats = 1, 
                                   classProbs = TRUE, 
                                   summaryFunction = twoClassSummary, 
                                   search = "random")

rF_model_check <- train(Churn ~., data = train_data, method = "rf", 
                        ntree = 2001, tuneLength = 3,
                        metric = "ROC", trControl = simulation_control)

rF_model_check

# It can be observed that the search method gives an indication of an optimal 
# Mtry = 3. The ntree parameter is set at 2001 to allow for increased 
# accuracy, since the Dataset is relatively minimal. We can inform the choice 
# of this ntree by observing the plot of the Random Forest model.

set.seed(123)
rF_model <- randomForest(Churn ~., data = train_data, mtry = 3, ntree = 2001, 
                         importance = TRUE)
plot(rF_model)

# The graph informs that the error margin tends to flatten out after around 500 
# decision trees. This allows the reduction of the overall number of decision 
# trees employed, which saves computation time with negligible loss of accuracy.

# Confusion matrix of current model

rF_model    # The result is a confusion matrix.

# Now, re-creating the model with significantly reduced number of trees.

set.seed(123)
rF_model <- randomForest(Churn ~., data = train_data, mtry = 3, ntree = 100, 
                         importance = TRUE)
rF_model    # Checking the confusion matrix

# The OOB error estimate is still approximately the same, which saves 
# computation without sacrificing efficiency.
# 
# Now that the model has been trained, we can validate the model using the 
# testing data.

prediction <- predict(rF_model, test_data, type = "response")
conf_mat <- confusionMatrix(prediction, test_data$Churn)
conf_mat

# The confusion matrix informs that the Random Forest, like the other algorithms 
# is better at classifying "NO" than "YES". Its overall accuracy is at 79.57%

# Other useful stats

all_errors <- unlist(errors(conf_mat))

randForest_overall_error <- all_errors[1]
randForest_type1_error <- all_errors[2]
randForest_type2_error <- all_errors[3]

cat(paste0("\nOverall error: ",randForest_overall_error,"%\n Type 1 error: ", 
    randForest_type1_error,"%\n Type 2 error: ",randForest_type2_error,"%\n"))

# The variable importance ranking of the Random Forest can also observed

varImpPlot(rF_model, sort=TRUE, n.var = 5, main = 'Top 5 Features/Predictors')

# Just as seen in previous machine learning approaches, the resulting chart 
# informs that the most important features/predictors are "Tenure, Monthly 
# charges, contract status and Total Charges". This can be useful in advising 
# the Telecoms company to pay special attention to customers who have not 
# committed to a long term contract and are likely to have a shorter tenure 
# and reduced monthly/total charges. Since Internet service is also a top 
# predictor of churn, incentives can be offered in form of internet stability, 
# increased bandwidth, etc. to motivate loyalty and commitment to a longer 
# contract/tenure.

#_______________________________________________________________________________
################################################################################
#                             4.0. CONCLUSION 
#_______________________________________________________________________________

# The Telecoms Dataset has been explored for predictors that inform of possible 
# causes for the observed problem of customer churn. The application of three 
# machine learning techniques to the Telecoms Dataset provided interesting 
# insights into the data.  
# Overall, of the three applied machine learning techniques, the Logistic 
# regression approach provided the best predictions and accuracy: least error 
# rate and type-1 error. 
# As expected, the Random Forest algorithm had a slightly improved performance 
# over the decision tree. These can be seen in the dataframe below:

# column and row names for the dataFrame
colnames <- c("Overall Error Rate", "Type-1 Error Rate", 
              "Type-2 Error Rate")
rownames <- c("Logistic Regression","Decision Trees", "Random Forest")

# Data Frame for comparison

logistic_regression <- c(logisticReg_overall_error,logisticReg_type1_error,
                         logisticReg_type2_error)
decision_tree <- c(dectree_overall_error,dectree_type1_error,
                   dectree_type2_error)
randomForest_tree <- c(randForest_overall_error,randForest_type1_error,
                       randForest_type2_error)

comp_table <- data.frame(rbind(logistic_regression, decision_tree, 
              randomForest_tree), row.names = rownames)
colnames(comp_table) <- colnames
comp_table

# Finally, the three algorithms had a consensus on the top
# predictors for customer churn based on the Telecoms Dataset: Monthly charges, 
# Total charges, and Tenure.

# Debugged !!!




