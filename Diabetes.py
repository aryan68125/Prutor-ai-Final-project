#!/usr/bin/env python
# coding: utf-8

# # This Machine Learning model will detect if a person have diabetes or not

# ## Objective : 
# ### In this we have to make a machine learning model that is capable to identify between those people that have diabetes and those who do not have diabetes.

# ## Models used for testing :
# ### In this stage we are applying SVM , Logistic Regression, Polynomial Regression, Linear Regression models to test which model gives us best accuracy score with least number of miss classified samples from the diabetes data Frame.

# ![](framework.png)

# ## Importing the required python libraries

# In[221]:


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


# In[222]:


### read csv file
df = pd.read_csv('diabetes.csv')


# In[223]:


df


# In[224]:


df.shape #checking the shape of the dataframe


# In[225]:


# gives information of the data types
df.info


# In[226]:


df.isnull().sum() # checking if the dataframe has any null value or not 


# In[227]:


# basic statistic details about the data (note only numerical columns would be displayed here unless parameter 
# include="all")
df.describe()


# In[228]:


#print a graph for correlation
plt.figure(figsize=(8,8))
plt.title('Correlation between dataset variables')
sb.heatmap(df.corr(), annot=True)


# In[229]:


df.Outcome.value_counts() # getting the number of classes in the target columns


# In[230]:


x = df.iloc[:,:-1] # features


# In[231]:


y = df.iloc[:,8] #target values


# In[232]:


np.bincount(y)


# ## Applying SVM on the dibetes dataFrame

# In[233]:


xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=.2,random_state=12)


# In[234]:


p = [{'kernel':['linear'],'C':[.01,1,10]},{'kernel':['rbf'],'gamma':[.001,.01,.1,10]}]
clf = GridSearchCV(SVC(),param_grid=p,cv=5,scoring='accuracy')
clf.fit(xtrain,ytrain)


# In[235]:


clf.best_score_ #getting the best model with the highest score


# In[236]:


clf.best_params_ # getting the best model with the best hyper parameters with the highest score


# In[237]:


clf1 = clf.best_estimator_ # saving the model with the best hyper parameter in the clf variable
clf1


# In[238]:


pred = clf1.predict(xtest)


# In[239]:


from sklearn.metrics import accuracy_score
accuracy_score(ytest,pred)


# ## Plotting confusion matrix

# In[240]:


c1 = confusion_matrix(ytest,pred)
c1


# In[241]:


np.bincount(ytest) # it returns the number of samples in each class


# In[242]:


recall_score(ytest,pred)


# In[243]:


precision_score(ytest,pred)


# ## now testing the model

# In[244]:


pred_train = clf1.predict(xtrain)


# In[245]:


np.where(pred_train!=ytrain) #getting all the missclassified samples in train dataset


# In[246]:


pred_test = clf1.predict(xtest)
pred_test


# In[247]:


np.where(pred_test!=ytest) #getting all the missclassified samples in test dataset


# ### Conclusion SVm model have the accuracy score of 81%

# ## Now Applying Logistic Regression on Dibetes Dataframe

# In[248]:


x # x has all the features 


# In[249]:


y # y has all the values of the target columns


# ## Splitting data into train and test dataset

# In[250]:


xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=.2,random_state=12)


# In[251]:


xtrain.shape


# In[252]:


xtest.shape


# In[253]:


ytrain.shape


# In[254]:


ytest.shape


# ### Transforming data using Feature Scaling (MinMaxScaler) OR Normalizing data using MinMax Scalar. It is very important to normalize data in the dataframe before feeding it to Logistic regression

# In[255]:


from sklearn.preprocessing import MinMaxScaler 
minmax_scaler = MinMaxScaler() # initialization of minmaxScalar


# In[256]:


minmax_scaler_train = minmax_scaler.fit_transform(xtrain)
minmax_scaler_test = minmax_scaler.transform(xtest)


# ### Use PCA ->It reduces dimension or Features with the minimum loss of information to reduce model training time and remove less important features in the dataset

# In[257]:


from sklearn.decomposition import PCA
pca = PCA(n_components=.95)


# In[258]:


pca_train = pca.fit_transform(minmax_scaler_train)
pca_test = pca.transform(minmax_scaler_test)


# In[259]:


pca_train.shape


# In[260]:


pca_test.shape


# ### Now applying Logistic Regression

# In[261]:


from sklearn.linear_model import LogisticRegression
log = LogisticRegression(multi_class='multinomial',max_iter=10000)


# In[262]:


log.fit(pca_train,ytrain)


# In[263]:


pred = log.predict(pca_test)
pred


# In[264]:


ytest


# In[265]:


score = log.score(pca_test,ytest)
score


# ### Conclusion Logistic regression with PCA and minMaxScalar transformation gives accuracy score of 80%

# ## Now what if we remove PCA and use Logistic regression directply after normalizing data using MinMax scalar

# In[266]:


log.fit(minmax_scaler_train,ytrain)


# In[267]:


pred = log.predict(minmax_scaler_test)
pred


# In[268]:


score = log.score(minmax_scaler_test,ytest)
score


# ### So if we do not use PCA the accuracy score goes from 80 -> 81%

# ### What if we use standar scalar instead of minMax Scalar

# In[269]:


from sklearn.preprocessing import StandardScaler
std = StandardScaler()


# In[270]:


std_train = std.fit_transform(xtrain)
std_test = std.transform(xtest)


# In[271]:


log.fit(std_train,ytrain)


# In[272]:


pred = log.predict(std_test)


# In[273]:


score = log.score(std_test,ytest)
score


# ### There is no change in the Logistic regression score even if we use standard scalar

# ### what if we use Logistic regression directly without any normalization

# In[274]:


log.fit(xtrain,ytrain)


# In[275]:


pred = log.predict(xtest)


# In[276]:


score = log.score(xtest,ytest)


# In[277]:


score


# ### Still no improvement in the model accuracy even if we use Logistic regression directly

# ## Now Applying Polynomial Regression on Dibetes Dataset

# In[278]:


# SimpleRegression => Simple Linear regression and multi Linear regression
# If r2_score is not close to 1 then Linear regression is not good for the the dataset
# Here now we have to use different Regression Model (Polynomial Regression)

# Polynomial Regression = Polynomial Features + Linear Regression 
#It's function is to transform data


# In[279]:


from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2) 
poly_x_train = poly.fit_transform(xtrain) # Tranforming x (input data) OR here in this case train_input


# In[280]:


poly_x_train.shape


# In[281]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression() # Initializing Linear regression


# In[282]:


lr.fit(poly_x_train,ytrain) # training the Linear regression model after polynomial transformation of data


# In[283]:


from sklearn.metrics import r2_score,mean_squared_error #testing model accuracy
pred_train = lr.predict(poly_x_train) # Running Predictions on train dataset


# In[284]:


score_train = r2_score(ytrain,pred_train) # scoring our Polynomial regression model
score_train


# In[285]:


# Just like what we did with our training data we need to transform the test input 
poly_x_test = poly.transform(xtest)


# In[286]:


pred_test = lr.predict(poly_x_test) #Running prediction on test dataset
score_test = r2_score(ytest,pred_test)
score_test


# ### So here we can see that polynomial transformation of data before applying Linear Regression did not give us a model with a good score that means here Polynomial Transformation of data will not work

# ## Now applying Raw Linear Regression model on diebetes dataset

# In[287]:


lr.fit(xtrain,ytrain)


# In[288]:


predict_test = lr.predict(xtest)


# In[289]:


score_test = r2_score(ytest,predict_test)
score_test


# In[290]:


predict_train = lr.predict(xtrain)


# In[291]:


score_train = r2_score(ytrain,predict_train)
score_train


# ### Linear Regression Model is also not suitable for classifiying diebets patient in the diebetes dataset

# ### Hence according to my analysis SVM and Logisitc Regression are the only two Machine Learning model with the descent score of 81% and can be used in the dibetes dataset.
