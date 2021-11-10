#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import gensim
from gensim import corpora, models
from numpy import mean
from numpy import std
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from skmultilearn.problem_transform import LabelPowerset
from sklearn.naive_bayes import GaussianNB
from sklearn.multioutput import ClassifierChain


# # Data and Notebook Setup

# In[ ]:


df = pd.read_csv('IMA_recommendation_simulation_data.csv')


# In[ ]:


print("Length: ",len(df))
size = df.size
shape = df.shape
print("Size of DataSet: ", size)
print("Shape of DataSet: ", shape)
print("List of Data types:\n", df.dtypes)


# The shape of this dataset is (161563, 16), and there are 7 unique Condition categories.

# In[ ]:


print("Number of Condition Categories: ", df['CurrentCondition'].value_counts().count())


# # Exploratory Data Analysis

# In[ ]:


print("Missingness: \n", df.isnull().sum())


# The following columns have missing data: order_distance, order_origin_weight, rate_norm, est_cost_norm, and CurrentCondtion. The request ID is unique because it is a type object. 

# In[ ]:


df.corr(method='pearson')


# In[ ]:


df.corr().unstack().sort_values(ascending = False).drop_duplicates()


# In[ ]:


sn.heatmap(df.corr())


# In[ ]:


plt.scatter(df.order_distance, df.miles)
plt.show()


# This scatterplot of order distance plotted against miles, shows an obvious positive relationship. There is a steady incline, showing that as the order distance increases, the miles increase at about the same rate. This would make sense seeing that there is a very large correlation value of 0.982291.

# In[ ]:


plt.scatter(df.miles, df.est_cost_norm)
plt.show()


# This scatterplot of miles plotted against estimated cost (normalized), shows a steep positive relationship. There is a large incline, showing that as the miles increase, normalized estimated cost increases at a very rapid rate. The correlation value of 0.742555 shows that these variables overlap much more than others.

# In[ ]:


print(df.CurrentCondition.value_counts(normalize = True))
df.CurrentCondition.value_counts().plot.barh()
plt.show()


# This bar graph visualization shows how many orders are in each category of Current Condition. From the bars, we can see that most orders are Accepted, with Rejected being the second most likely. 

# In[ ]:


result = pd.pivot_table(data=df, index='CurrentCondition', columns='color',values='est_cost_norm')
print(result)
sn.heatmap(result, annot=True, cmap = 'RdYlGn', center=0)
plt.show()


# This visualization gives a table as well as a heat map to explain what is going on. The boxes with darker red show that there is a lower estimated cost, while those with the green are at a higher estimated cost. The boxes themselves are fit to be in a category of the Current Condition and a category of the color given as well. 

# # Classification Model 

# In[ ]:


df = df.drop('request_id', axis=1)
df = df.iloc[: , 1:]


# In[ ]:


df = df[df['CurrentCondition'].notna()]
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
categ = ['order_equipment_type', 'weekday', 'CurrentCondition', 'color']
df[categ] = df[categ].apply(le.fit_transform)


# #### Processing the data with imputation

# In[ ]:


df['order_origin_weight'].fillna(df['order_origin_weight'].mean(), inplace = True)
df['rate_norm'].fillna(df['rate_norm'].mean(), inplace = True)
df['est_cost_norm'].fillna(df['est_cost_norm'].mean(), inplace = True)
df['order_distance'].fillna(df['order_distance'].mean(), inplace = True)


# In[ ]:


X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values


# #### Train and Test sets

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)


# #### Decision Tree Classifier

# In[ ]:


from sklearn import tree 
variables = df.iloc[:,:-1]
results = df.iloc[:,-1]

decision_tree = tree.DecisionTreeClassifier().fit(X_train, y_train)
tree_predictions = decision_tree.predict(X_test)
tree_expected = y_test


# In[ ]:


matches = (tree_predictions == tree_expected)
print("Correct Matches:", matches.sum())
print("Data Points:", len(matches))
print("Accuracy Rate:", matches.sum() / float(len(matches)))


# In[ ]:


from sklearn import metrics
print(metrics.classification_report(tree_predictions,tree_expected))


# In[ ]:


print(metrics.confusion_matrix(tree_predictions,tree_expected))
plt.show()


# In[ ]:


from sklearn.inspection import permutation_importance
imps = permutation_importance(decision_tree, X_test, y_test)
tree_values= imps.importances_mean
print(tree_values)


# In[ ]:


names = list(df)
del names[-1]


# In[ ]:


data = {'names': names, 'values': tree_values}
Table_tree = pd.DataFrame(data=data)
print(Table_tree.sort_values(by='values', ascending=False))


# In[ ]:


ax = Table_tree.plot.barh(x='names', y='values')
ax.set_ylabel(None)


# In[ ]:


from sklearn.model_selection import cross_val_score
print(cross_val_score(decision_tree, X, y, cv=3))


# This classification model is better to produce the Current Condition of a load. The model would place the load correctly  70.73% of the time. This is a decent way to predict a load. The geographic region and weight of the shipment are the best features of this model. The graph above explains the features with the most importance as having the highest bar graph. This model is slightly overfit, seeing that there are different mean scores.

# #### Gaussian Naive Bayes Model

# In[ ]:


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB().fit(X_train, y_train)
gnb_predictions = gnb.predict(X_test)
gnb_expected = y_test


# In[ ]:


matches = (gnb_predictions == gnb_expected)
print("Correct Matches:", matches.sum())
print("Data Points:", len(matches))
print("Accuracy Rate:", matches.sum() / float(len(matches)))


# In[ ]:


from sklearn import metrics
print(metrics.classification_report(gnb_predictions,gnb_expected))


# In[ ]:


print(metrics.confusion_matrix(gnb_predictions,gnb_expected))
plt.show()


# In[ ]:


from sklearn.metrics import mean_squared_error
mean_squared_error(gnb_predictions,gnb_expected)


# In[ ]:


from sklearn.inspection import permutation_importance
imps = permutation_importance(gnb, X_test, y_test)
values_gnb = list(imps.importances_mean)


# In[ ]:


data = {'names': names, 'values': values_gnb}
Table_gnb = pd.DataFrame(data=data)
print(Table_gnb.sort_values(by='values', ascending=False))


# In[ ]:


ax = Table_gnb.plot.barh(x='names', y='values')
ax.set_ylabel(None)


# In[ ]:


from sklearn.model_selection import cross_val_score
print(cross_val_score(gnb, X, y, cv=3))


# This classification model is correct 53.73% of the time about the condition of a load. I would not recommend this model to predict the status of a load, seeing that it just barely over 50%. The most important features of this model are the shipment weight and the number of days before pickup date that the customer placed the order. Because of the error rate, we cannot confirm if these would even be the best features to use. This model is very overfit becasue of the large difference in mean scores. This model overall needs to be refit to be of more use. 

# # Context and Critical Reflection

# ### I have a few questions about this dataset that would have helped me to understand this analysis further. Who assigns the color of the order, is it automatic or manual (up for interpretation)? Does the type of truck requested rely on the weight of the load?
# 
# In order to answer these questions, I would look to ask the team that assigns the color, possibly the vendor? I would aslo seek out who chooses the trucks for each load and determine if there is a cut off weight to choose equipment types. 
