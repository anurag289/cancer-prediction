#!/usr/bin/env python
# coding: utf-8

# In[31]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.preprocessing import MinMaxScaler


# In[3]:


wisc = pd.read_csv('wisc_bc_data.csv')
wisc.head()


# In[4]:


#checking how the value are behaving with the statistics
wisc.describe()


# In[7]:


wisc.columns


# In[9]:


wisc.diagnosis.value_counts()


# In[16]:


#plotting for all the mean values for both the diagnosis

column_mean=['radius_mean', 'texture_mean', 'perimeter_mean','area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']
for col in column_mean:
    plt.scatter(range(357),col,data=wisc[wisc.diagnosis=='B'],s=5,marker='*',label='Benign',c='green')
    plt.scatter(range(212),col,data=wisc[wisc.diagnosis=='M'],s=5,marker='^',label='Malignant',c='red')
    plt.legend()
    plt.xlabel('Index')
    plt.ylabel(col)
    plt.show()


# In[17]:


#plotting for all the worst values for both the diagnosis

column_mean=['radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst']
for col in column_mean:
    plt.scatter(range(357),col,data=wisc[wisc.diagnosis=='B'],s=5,marker='*',label='Benign',c='green')
    plt.scatter(range(212),col,data=wisc[wisc.diagnosis=='M'],s=5,marker='^',label='Malignant',c='red')
    plt.legend()
    plt.xlabel('Index')
    plt.ylabel(col)
    plt.show()


# In[18]:


#plotting for all the se values for both the diagnosis

column_mean=['radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se']
for col in column_mean:
    plt.scatter(range(357),col,data=wisc[wisc.diagnosis=='B'],s=5,marker='*',label='Benign',c='green')
    plt.scatter(range(212),col,data=wisc[wisc.diagnosis=='M'],s=5,marker='^',label='Malignant',c='red')
    plt.legend()
    plt.xlabel('Index')
    plt.ylabel(col)
    plt.show()


# In[23]:


wisc=wisc[['diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst']]

wisc.head()
wisc.corr()


# In[26]:


import seaborn as sns
sns.pairplot(wisc)


# In[27]:


#get correlations of each features in dataset
corrmat = wisc.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(wisc[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[37]:


X=wisc[[ 'radius_mean','perimeter_mean']]
y= wisc['diagnosis']


# In[65]:


Scaler = MinMaxScaler()
#X = pd.DataFrame(Scaler.fit_transform(X),columns=['radius_mean','perimeter_mean'])
#X_train,X_test,y_train,y_test  = train_test_split(X,y,test_size=0.2,stratify=y)
#print(y_train.value_counts())
#accuracy=[]
r = range(1,20,2)

for z in range(0,10):
    X = pd.DataFrame(Scaler.fit_transform(X),columns=['radius_mean','perimeter_mean'])
    X_train,X_test,y_train,y_test  = train_test_split(X,y,test_size=0.2,stratify=y)
    print(y_train.value_counts())
    accuracy=[]
    for k in r:
        clf = KNeighborsClassifier(n_neighbors=k,weights='distance')
        clf.fit(X_train,y_train)
        print("for the neighbiur choosen as",k,"the accuracy is")
        print(clf.score(X_test,y_test))
        accuracy.append(clf.score(X_test,y_test))

    plt.plot(r,accuracy,marker='o')
    plt.xlabel("Neighbours")
    plt.ylabel("Accuracy")
    print("\nIteration", z+1)
    plt.title("KNN with scaling, startify, weights =distance and formula used in euclidiean the iteration is ")
    plt.show()
    accuracy=accuracy.clear()

for z in range(0,10):
    X = pd.DataFrame(Scaler.fit_transform(X),columns=['radius_mean','perimeter_mean'])
    X_train,X_test,y_train,y_test  = train_test_split(X,y,test_size=0.2,stratify=y)
    print(y_train.value_counts())
    accuracy=[]
    r = range(1,20,2)
    for k in r:
        clf = KNeighborsClassifier(n_neighbors=k,weights='distance',p=1)
        clf.fit(X_train,y_train)
        print("for the neighbiur choosen as",k,"the accuracy is")
        print(clf.score(X_test,y_test))
        accuracy.append(clf.score(X_test,y_test))
    
    plt.plot(r,accuracy,marker='o')
    print("\nIteration", z+1)
    plt.xlabel("Neighbours")
    plt.ylabel("Accuracy")
    plt.title("KNN with scaling, startify, weights =distance and formula used in manahatten")
    plt.show()


# # # From the above iteration we can classify that with manahatten with neighbours we are getting the max accuracy multiple time.
# 
# # # # So, I am creating a model with that

# In[130]:


X_train,X_test,y_train,y_test  = train_test_split(X,y,test_size=0.2,stratify=y)
clf = KNeighborsClassifier(n_neighbors=7,weights='distance',p=1)
clf.fit(X_train,y_train)
print(clf.score(X_test,y_test))


# In[131]:


import pickle
#opening the file where I want to store the data; wb is write byte mode
file=open('Knearforcancer.pkl','wb')

#dummp the info in that file
pickle.dump(clf, file)


# # As the previous one is pretty difficult to decide trying to make some estmation  created a list which stores all the values

# In[142]:


Scaler = MinMaxScaler()
#X = pd.DataFrame(Scaler.fit_transform(X),columns=['radius_mean','perimeter_mean'])
#X_train,X_test,y_train,y_test  = train_test_split(X,y,test_size=0.2,stratify=y)
#print(y_train.value_counts())
#accuracy=[]
filldata=[['Algorithm','Accuracy','neighbours']]
r = range(1,20,2)

for z in range(0,10):
    X = pd.DataFrame(Scaler.fit_transform(X),columns=['radius_mean','perimeter_mean'])
    X_train,X_test,y_train,y_test  = train_test_split(X,y,test_size=0.2,stratify=y)
    print(y_train.value_counts())
    accuracy=[]
    for k in r:
        clf = KNeighborsClassifier(n_neighbors=k,weights='distance')
        clf.fit(X_train,y_train)
        print("for the neighbiur choosen as",k,"the accuracy is")
        print(clf.score(X_test,y_test))
        accuracy.append(clf.score(X_test,y_test))
        filldata.append([["Euclidiean",clf.score(X_test,y_test),k]])

    plt.plot(r,accuracy,marker='o')
    plt.xlabel("Neighbours")
    plt.ylabel("Accuracy")
    print("\nIteration", z+1)
    plt.title("KNN with scaling, startify, weights =distance and formula used in euclidiean the iteration is ")
    plt.show()
    accuracy=accuracy.clear()
    
for z in range(0,10):
    X = pd.DataFrame(Scaler.fit_transform(X),columns=['radius_mean','perimeter_mean'])
    X_train,X_test,y_train,y_test  = train_test_split(X,y,test_size=0.2,stratify=y)
    print(y_train.value_counts())
    accuracy=[]
    for k in r:
        clf = KNeighborsClassifier(n_neighbors=k,weights='distance',p=1)
        clf.fit(X_train,y_train)
        print("for the neighbiur choosen as",k,"the accuracy is")
        print(clf.score(X_test,y_test))
        accuracy.append(clf.score(X_test,y_test))
        filldata.append([["Manhatten",clf.score(X_test,y_test),k]])

    plt.plot(r,accuracy,marker='o')
    plt.xlabel("Neighbours")
    plt.ylabel("Accuracy")
    print("\nIteration", z+1)
    plt.title("KNN with scaling, startify, weights =distance and formula used in manhatten the iteration is ")
    plt.show()
    accuracy=accuracy.clear()


# In[143]:


#printing the created list we can even add iteration but I did not
filldata


# In[153]:


#pushing the created list to excel for better understanding
pd.DataFrame(filldata).to_excel('output.xlsx', header=False, index=False)


# In[177]:


X=wisc[[ 'radius_mean','perimeter_mean']]
y= wisc['diagnosis']

print(X.head())
print(y.head())
X_train,X_test,y_train,y_test  = train_test_split(X,y,test_size=0.2,stratify=y)
clf = KNeighborsClassifier(n_neighbors=7,weights='distance',p=1)
clf.fit(X_train,y_train)
print(clf.score(X_test,y_test))
print(X_train.head())
print(y_train.head())


# In[178]:


import pickle
#opening the file where I want to store the data; wb is write byte mode
file=open('Knearforcancer.pkl','wb')

#dummp the info in that file
pickle.dump(clf, file)


# In[180]:


prediction=clf.predict([[14.870,96.12]])
output=prediction[0]
output


# In[ ]:




