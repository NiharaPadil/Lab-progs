#!/usr/bin/env python
# coding: utf-8

# In[2]:


#1

import pandas as pd
df = pd.read_csv('enjoysport.csv')
#df = df.drop(['slno'],axis=1)
column_length = df.shape[1]
df.head()

h = ['0']*(column_length-1)
hp =[]
hn =[]


for training_example in df.values:
    if training_example[-1] != 'no':
        hp.append(list(training_example))
    else:
        hn.append(list(training_example))   



for i in range(len(hp)):
    for j in range(column_length - 1):
        if (h[j]=="0"):
            h[j]=hp[i][j]
        if (h[j]!=hp[i][j]):
            h[j]='?'
        else:
            h[j]=hp[i][j]

print(f'Postive hypothesis:\n{hp}')
print(f'Negative hypothesis:\n{hn}')
print(f'Maximlly Specific hypothesis:\n{h}')



# In[3]:


#2

import numpy as np
import pandas as pd

data = pd.DataFrame(data=pd.read_csv('enjoysport.csv'))

concepts = np.array(data.iloc[:, 0:-1])
print("Concepts:\n", concepts)

target = np.array(data.iloc[:, -1])
print("Target:\n", target)

def learn(concepts, target):
    specific_h = concepts[0].copy()
    print("Initialization of specific_h and general_h")
    print("Specific_h:", specific_h)
    
    general_h = [["?" for i in range(len(specific_h))] for i in range(len(specific_h))]
    print("General_h:", general_h)
    
    # Training the model
    for i, h in enumerate(concepts):
        if target[i] == "yes":
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    specific_h[x] = '?'
                    general_h[x][x] = '?'
            print(f"Specific_h after instance {i+1}:", specific_h)
        
        if target[i] == "no":
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    general_h[x][x] = specific_h[x]
                else:
                    general_h[x][x] = '?'
            print(f"General_h after instance {i+1}:", general_h)
    
    indices = [i for i, val in enumerate(general_h) if val == ['?' for _ in range(len(specific_h))]]
    for i in indices:
        general_h.remove(['?' for _ in range(len(specific_h))])
    
    return specific_h, general_h

s_final, g_final = learn(concepts, target)
print("Final Specific_h:\n", s_final)
print("Final General_h:\n", g_final)


# In[5]:


#3

import pandas as pd
from sklearn .preprocessing import MinMaxScaler

data=pd.read_csv('student.csv')

print(data)

data=data.dropna()

data=data.drop_duplicates(subset='Name')

print('\n',data)

scaler=MinMaxScaler()

scaled_data = scaler.fit_transform(data[['Age','GPA']])

data[['Age','GPA']] = scaled_data

data['Student_Info'] = data['Name'] + '(' + data['Grade'] + ')'

print('\n',data.head())


# In[7]:


#4

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target
X = X[y != 2]
y = y[y != 2]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)
y_pred = svm_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k')
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T

Z = svm_classifier.decision_function(xy).reshape(XX.shape)

ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
ax.scatter(svm_classifier.support_vectors_[:, 0], svm_classifier.support_vectors_[:, 1], s=100, linewidth=1, facecolors='none', edgecolors='k')

plt.xlabel('Feature1')
plt.ylabel('Feature2')
plt.title('SVM Binary Classifier')

plt.show()


# In[8]:


#5

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Reading the dataset
dataset = pd.read_csv('PlayTennis.csv')

# Defining features and target variable
features = ['Outlook', 'Temperature', 'Humidity', 'Wind']
X = dataset[features]
Y = dataset['PlayTennis']

# Encoding categorical variables
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_encoded = pd.DataFrame(encoder.fit_transform(X), columns=encoder.get_feature_names_out(features))

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X_encoded, Y, test_size=0.30, random_state=100)

# Building the decision tree
dtree = DecisionTreeClassifier(criterion="entropy", random_state=100)
dtree.fit(X_train, y_train)
y_pred = dtree.predict(X_test)

# Classifying the new instance based on the training data
def classify_new_instance(outlook, temperature, humidity, wind, encoder):
    instance = [[outlook, temperature, humidity, wind]]
    instance_df = pd.DataFrame(instance, columns=features)
    instance_encoded = encoder.transform(instance_df)
    feature_names = encoder.get_feature_names_out(features)
    instance_encoded_df = pd.DataFrame(instance_encoded, columns=feature_names)
    prediction = dtree.predict(instance_encoded_df)
    return prediction[0]

# Predicting the class of a new instance
pred = classify_new_instance("Rain", "Mild", "High", "Strong", encoder=encoder)
print("Prediction:", pred)

# Evaluating the model
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))


# In[9]:


#6

# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap

data_set= pd.read_csv('user_data.csv')

x= data_set.iloc[:, [2,3]].values
y= data_set.iloc[:, 4].values

x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.25, random_state=0)

st_x= StandardScaler()
x_train= st_x.fit_transform(x_train)
x_test= st_x.transform(x_test)

classifier= RandomForestClassifier(n_estimators= 10, criterion="entropy")
classifier.fit(x_train, y_train)

y_pred= classifier.predict(x_test)

cm= confusion_matrix(y_test, y_pred)

x_set, y_set = x_test, y_test
x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1,step =0.01),np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))
plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape), alpha = 0.75, cmap = ListedColormap(('purple','green' )))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],c = ListedColormap(('purple', 'green'))(i), label = j)

plt.title('Random Forest Algorithm(Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


# In[12]:


#7

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import pandas as pd

# Reading the dataset
msg = pd.read_csv("naivetext.csv", names=['message', 'label'])
msg['labelnum'] = msg.label.map({'pos': 1, 'neg': 0})

# Defining features and target variable
X = msg.message
y = msg.labelnum

# Splitting the dataset into train and test data
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=42)

# Output of the words or Tokens in the text documents
count_vect = CountVectorizer()
xtrain_dtm = count_vect.fit_transform(xtrain)
xtest_dtm = count_vect.transform(xtest)

print('\nThe words or Tokens in the text documents\n')
# If get_feature_names_out() gives an error, then replace it with get_feature_names()
print(count_vect.get_feature_names_out())
df = pd.DataFrame(xtrain_dtm.toarray(), columns=count_vect.get_feature_names_out())

# Training Naive Bayes (NB) classifier on training data
clf = MultinomialNB().fit(xtrain_dtm, ytrain)
predicted = clf.predict(xtest_dtm)

# Printing accuracy, Confusion matrix, Precision and Recall
print('\nAccuracy of the classifier is', metrics.accuracy_score(ytest, predicted))
print('\nConfusion matrix')
print(metrics.confusion_matrix(ytest, predicted))
print('\nThe value of Precision', metrics.precision_score(ytest, predicted))
print('\nThe value of Recall', metrics.recall_score(ytest, predicted))


# In[11]:


#8

import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

df = pd.read_csv("pima_indian.csv")
feature_col_names = ['num_preg', 'glucose_conc', 'diastolic_bp', 'thickness', 'insulin', 'bmi', 'diab_pred', 'age']
predicted_class_names = ['diabetes']

X = df[feature_col_names].values 
y = df[predicted_class_names].values 

xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.33)

print('\n The total number of Training Data:', ytrain.shape) 
print('\n The total number of Test Data:', ytest.shape)

# Training Naive Bayes (NB) classifier on training data. 
clf = GaussianNB().fit(xtrain, ytrain.ravel())

predicted = clf.predict(xtest)
predictTestData = clf.predict([[6,148,72,35,0,33.6,0.627,50]])

# Printing Confusion matrix, accuracy, Precision and Recall 
print('\n Confusion matrix') 
print(metrics.confusion_matrix(ytest, predicted))

print('\n Accuracy of the classifier is', metrics.accuracy_score(ytest, predicted)) 
print('\n The value of Precision', metrics.precision_score(ytest, predicted)) 
print('\n The value of Recall', metrics.recall_score(ytest, predicted)) 
print("Predicted Value for individual Test Data:", predictTestData)


# In[13]:


#9

import numpy as np
import pandas as pd
import csv
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.models import BayesianModel
from pgmpy.inference import VariableElimination

heartDisease = pd.read_csv('heart.csv')
heartDisease = heartDisease.replace('?', np.nan)

model = BayesianModel([
    ('age', 'heartdisease'),
    ('sex', 'heartdisease'),
    ('exang', 'heartdisease'),
    ('cp', 'heartdisease'),
    ('heartdisease', 'restecg'),
    ('heartdisease', 'chol')
])

print('\nLearning CPD using Maximum likelihood estimators')
model.fit(heartDisease, estimator=MaximumLikelihoodEstimator)

print('\nInferencing with Bayesian Network:')
HeartDiseasetest_infer = VariableElimination(model)

print('\n1. Probability of HeartDisease given evidence= restecg')
q1 = HeartDiseasetest_infer.query(variables=['heartdisease'], evidence={'restecg': 1})
print(q1)

print('\n2. Probability of HeartDisease given evidence= cp')
q2 = HeartDiseasetest_infer.query(variables=['heartdisease'], evidence={'cp': 2})
print(q2)


# In[14]:


from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
dataset=load_iris()
X_train,X_test,y_train,y_test=train_test_split(dataset["data"],dataset["target"],random_state=0)
kn=KNeighborsClassifier(n_neighbors=3)
kn.fit(X_train,y_train)
KNeighborsClassifier(n_neighbors=3)
prediction=kn.predict(X_test)
confusion_matrix(y_test,prediction)


# In[15]:


import numpy as np
import pandas as pd

# Load data from CSV into a DataFrame
data = pd.read_csv('enjoysport.csv')

# Extract concepts and target from the DataFrame
concepts = data.iloc[:, :-1].values  # Concepts are all columns except the last one
target = data.iloc[:, -1].values     # Target is the last column

def learn(concepts, target):
    # Initialize specific_h with the first instance of concepts
    specific_h = concepts[0].copy()
    # Initialize general_h with a list of "?" with the same length as specific_h
    general_h = [["?" for _ in range(len(specific_h))] for _ in range(len(specific_h))]
    
    # Iterate over each instance in concepts and update specific_h and general_h
    for i, h in enumerate(concepts):
        if target[i] == "yes":
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    specific_h[x] = '?'
                    general_h[x][x] = '?'
        
        if target[i] == "no":
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    general_h[x][x] = specific_h[x]
                else:
                    general_h[x][x] = '?'
    
    # Remove any incomplete general hypotheses
    general_h = [h for h in general_h if h != ['?' for _ in range(len(specific_h))]]
    
    return specific_h, general_h

# Call the learn function and print the final specific_h and general_h
s_final, g_final = learn(concepts, target)
print("Final Specific_h:\n", s_final)
print("Final General_h:\n", g_final)


# In[ ]:




