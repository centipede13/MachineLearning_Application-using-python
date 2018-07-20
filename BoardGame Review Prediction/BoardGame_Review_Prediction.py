# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 19:09:51 2018

@author: aaryan

Aim: To predict Game Review Ratings . From column names We Are Going To Predict Average_Rating 
     using LinearRegression and RandomForest.

"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


#Loading data
games = pd.read_csv("games.csv")

#printing the names of Column & total no of Rows/Columns
print(games.columns)
print(games.shape)

# Making HistoGram of all ratings in average_ratings column
plt.hist(games["average_rating"])
plt.show()

# Printing the firt row of all games with average_rating score=0 as a rating
print(games[games["average_rating"]==0].iloc[0])

# Printing the firt row of all games with average_rating score>0 as a rating
print(games[games["average_rating"]>0].iloc[0])

# We remove any rows without user reviews 
games = games[games["users_rated"]>0]
# We remove any row with missing values
#(dropna() 0 ->delete's rows with missing value & 1 ->delete's Columns with missing value)
games = games.dropna(axis=0)
# Makimg HistoGram Again
plt.hist(games["average_rating"])
plt.show()

# Develop a corelation matrix which tells us if there are some strong corelations b/w these parameters in our dataset
cor_mat = games.corr()
fig = plt.figure(figsize = (12,9))

sns.heatmap(cor_mat, vmax = .8, square = True)
plt.show()

# Getting column frome dataframe
columns = games.columns.tolist()
# Filtering the data that we do not want (like id etc..)
columns = [c for c in columns if c not in["bayes_average_rating", "average_rating", "type", "name", "id"]] 
# Storing the variable we will predict on 
target = "average_rating"

# Generate training set
train = games.sample(frac=0.8, random_state = 1)

#Selecting anything not in training set and putting it in test
test = games.loc[~games.index.isin(train.index)]

# Printing shapes
print(train.shape)
print(test.shape)

"""<--Importing and applying models from here-->"""

# Importing LinearRegression Model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Initializing Model Class
LR = LinearRegression()

# Fitting the Training Data with the Model
LR.fit(train[columns], train[target])

# Generating Predictions for our testing set
predictions = LR.predict(test[columns])

# Compute error b/w test predictions and actual values
mean_squared_error(predictions, test[target])

"""-- We found that MSE=2.078... ie greater than 0 therefore a linear model is not a good fit here 
    there fore we will now use the RandomForest Algorithm--
"""
# Importing RandomForest Model
from sklearn.ensemble import RandomForestRegressor

# Initializing the model
RFR = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=1)
 
# Fitting the training data
RFR.fit(train[columns], train[target])

# Generating predictions for our test set
predictions = RFR.predict(test[columns])

# Compute error b/w test predictions and actual values
mean_squared_error(predictions, test[target])

"""-- Now we check how accurate they are --"""
test[columns].iloc[0]

# Make Predictions With Both Models
avg_ratings_LR = LR.predict(test[columns].iloc[0].values.reshape(1, -1))
avg_ratings_RFR = RFR.predict(test[columns].iloc[0].values.reshape(1, -1))

# Printing The Predictions 
print(avg_ratings_LR)
print(avg_ratings_RFR)

# Printing Actual Value
print(test[target].iloc[0])

