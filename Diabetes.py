#!/usr/bin/env python
# coding: utf-8

# # This Machine Learning model will detect if a person have diabetes or not

# ![](framework.png)

# ## Importing the required python libraries

# In[36]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt #importing pyplot
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score
from sklearn.preprocessing import MinMaxScaler 
import seaborn as sb

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


# In[37]:


### read csv file
df = pd.read_csv('diabetes.csv')


# In[38]:


df


# In[39]:


df.shape #checking the shape of the dataframe


# In[40]:


# gives information of the data types
df.info


# In[62]:


df.isnull().sum() # checking if the dataframe has any null value or not 


# In[41]:


# basic statistic details about the data (note only numerical columns would be displayed here unless parameter 
# include="all")
df.describe()


# In[42]:


#print a graph for correlation
plt.figure(figsize=(8,8))
plt.title('Correlation between dataset variables')
sb.heatmap(df.corr(), annot=True)


# In[43]:


df.Outcome.value_counts() # getting the number of classes in the target columns


# In[44]:


x = df.iloc[:,:-1] # features


# In[45]:


y = df.iloc[:,8] #target values


# In[46]:


np.bincount(y)


# ## Applying SVM on the dibetes dataFrame

# In[47]:


svm = SVC(kernel = 'rbf', gamma=100)
svm.fit(x,y)


# In[48]:


pred = svm.predict(x)


# ## So here we can see on class 1 out of 500 samples there is 0 missclassified samples and in class 2 samples out of 268 smaples there are 0 missclassified samples

# In[49]:


c1 = confusion_matrix(y,pred)
c1


# In[51]:


np.bincount(y) # it returns the number of samples in each class


#  ### So in SVM model we do not need to use class_weight in the SVM model

# In[52]:


recall_score(y,pred)


# In[53]:


precision_score(y,pred)


# ## now testing the model

# ### splitting test and train data

# In[54]:


xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=.2,random_state=12)


# In[55]:


pred_train = svm.predict(xtrain)


# In[56]:


pred_train


# In[58]:


np.where(pred_train!=ytrain) #getting all the missclassified samples in train dataset


# In[59]:


pred_test = svm.predict(xtest)
pred_test


# In[60]:


np.where(pred_test!=ytest) #getting all the missclassified samples in test dataset


# ### as we can see here SVM model did not mis-classify any of the samples in the train and test dataset

# ## Now Applying Logistic Regression on Dibetes Dataframe

# In[63]:


x # x has all the features 


# In[64]:


y # y has all the values of the target columns


# ## Splitting data into train and test dataset

# In[65]:


xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=.2,random_state=12)


# In[66]:


xtrain.shape


# In[67]:


xtest.shape


# In[68]:


ytrain.shape


# In[69]:


ytest.shape


# ### Transforming data using Feature Scaling (MinMaxScaler) OR Normalizing data using MinMax Scalar. It is very important to normalize data in the dataframe before feeding it to Logistic regression

# In[70]:


from sklearn.preprocessing import MinMaxScaler 
minmax_scaler = MinMaxScaler() # initialization of minmaxScalar


# In[71]:


minmax_scaler_train = minmax_scaler.fit_transform(xtrain)
minmax_scaler_test = minmax_scaler.transform(xtest)


# ### Use PCA ->It reduces dimension or Features with the minimum loss of information to reduce model training time and remove less important features in the dataset

# In[72]:


from sklearn.decomposition import PCA
pca = PCA(n_components=.95)


# In[73]:


pca_train = pca.fit_transform(minmax_scaler_train)
pca_test = pca.transform(minmax_scaler_test)


# In[74]:


pca_train.shape


# In[75]:


pca_test.shape


# ### Now applying Logistic Regression

# In[76]:


from sklearn.linear_model import LogisticRegression
log = LogisticRegression(multi_class='multinomial',max_iter=10000)


# In[77]:


log.fit(pca_train,ytrain)


# In[78]:


pred = log.predict(pca_test)
pred


# In[79]:


ytest


# In[80]:


score = log.score(pca_test,ytest)
score


# ## Now what if we remove PCA and use Logistic regression directply after normalizing data using MinMax scalar

# In[81]:


log.fit(minmax_scaler_train,ytrain)


# In[82]:


pred = log.predict(minmax_scaler_test)
pred


# In[83]:


score = log.score(minmax_scaler_test,ytest)
score


# ### So if we do not use PCA the accuracy score goes from 80 -> 81%

# ### What if we use standar scalar instead of minMax Scalar

# In[85]:


from sklearn.preprocessing import StandardScaler
std = StandardScaler()


# In[86]:


std_train = std.fit_transform(xtrain)
std_test = std.transform(xtest)


# In[87]:


log.fit(std_train,ytrain)


# In[88]:


pred = log.predict(std_test)


# In[89]:


score = log.score(std_test,ytest)
score


# ### There is no change in the Logistic regression score even if we use standard scalar

# ### what if we use Logistic regression directly without any normalization

# In[90]:


log.fit(xtrain,ytrain)


# In[92]:


pred = log.predict(xtest)


# In[93]:


score = log.score(xtest,ytest)


# In[94]:


score


# ### Still no improvement in the model accuracy that means SVM has better accuracy than the Logistic Regression model so far

# ## Now Applying Polynomial Regression on Dibetes Dataset

# In[95]:


# SimpleRegression => Simple Linear regression and multi Linear regression
# If r2_score is not close to 1 then Linear regression is not good for the the dataset
# Here now we have to use different Regression Model (Polynomial Regression)

# Polynomial Regression = Polynomial Features + Linear Regression 
#It's function is to transform data


# In[96]:


from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2) 
poly_x_train = poly.fit_transform(xtrain) # Tranforming x (input data) OR here in this case train_input


# In[97]:


poly_x_train.shape


# In[98]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression() # Initializing Linear regression


# In[99]:


lr.fit(poly_x_train,ytrain) # training the Linear regression model after polynomial transformation of data


# In[100]:


from sklearn.metrics import r2_score,mean_squared_error #testing model accuracy
pred_train = lr.predict(poly_x_train) # Running Predictions on train dataset


# In[102]:


score_train = r2_score(ytrain,pred_train) # scoring our Polynomial regression model
score_train


# In[103]:


# Just like what we did with our training data we need to transform the test input 
poly_x_test = poly.transform(xtest)


# In[104]:


pred_test = lr.predict(poly_x_test) #Running prediction on test dataset
score_test = r2_score(ytest,pred_test)
score_test


# ### So here we can see that polynomial transformation of data before applying Linear Regression did not give us a model with a good score that means here Polynomial Transformation of data will not work

# ## Now applying Raw Linear Regression model on diebetes dataset

# In[105]:


lr.fit(xtrain,ytrain)


# In[107]:


predict_test = lr.predict(xtest)


# In[110]:


score_test = r2_score(ytest,predict_test)
score_test


# In[111]:


predict_train = lr.predict(xtrain)


# In[112]:


score_train = r2_score(ytrain,predict_train)
score_train


# ### Linear Regression Model is also not suitable for classifiying diebets patient in the diebetes dataset

# ### Hence according to my analysis SVM and Logisitc Regression are the only two Machine Learning model with the descent score and can be used in the dibetes dataset. Among the two SVM Model worked the best in this senario

# In[ ]:




