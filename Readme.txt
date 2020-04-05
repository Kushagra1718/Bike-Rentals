## CODE TESTED AND RUN ALMOST 5 TIMES BEFORE SUBMITTING. PLEASE WAIT FOR OUTPUT WHILE RUNNING 
## RANDOM FOREST FUNCTION  TAKES TIME TO IMPUTE AND SHOW RESULT. 






Steps to run Python code :-

Step - 1 : Install Anaconda Framework, with python 3.
Step - 2 : Open anaconda prompt, and install the required packages using the command 'pip install <required_package>' or from the shell.
Step - 3 : In the anaconda shell promt, change to the directory in which your python is located. 
Step - 4 : In the same prompt, enter 'jupyter notebook' to open the Python Jupyter Notebook.
Step - 5 : Once jupyter is open, direct to the folder in which the Python code with appropriate file name is present and double click.
Step - 6 : Press [ctrl + Enter] or [shift + Enter] to execute each cell or simply run by select and press enter key.
 
Important packages or libraries

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot  as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
import scipy.stats as stats
from scipy.stats import chi2_contingency
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices
from sklearn import metrics
from statsmodels.formula.api import ols
import statsmodels.api as sm
%matplotlib inline



# We can deploy the model either Online or Offline as per the requirement.




*************************************************************************************************************************
*************************************************************************************************************************
*************************************************************************************************************************

## CODE TESTED AND RUN ALMOST 5 TIMES BEFORE SUBMITTING. PLEASE WAIT FOR OUTPUT WHILE RUNNING 
## RANDOM FOREST FUNCTION  TAKES TIME TO IMPUTE AND SHOW RESULT. 



Steps to run R code :-
Step - 1 : Install RStudio.
Step - 2 : Install the required packages. Code: install.packages(<required_package>)
Step - 3 : Open the file "rpro1.R" in RStudio.
Step - 4 : Press [ctrl + Enter] or [shift + Enter] to execute each cell or simply run by select and press enter key.

Important packages or libraries

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