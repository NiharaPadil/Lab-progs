import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
dataset=pd.read_csv('Advertising.csv')
dataset.head(10)
dataset.shape
dataset.isna().sum()
dataset.duplicated().any()
fig,axs=plt.subplots(3,figsize=(5,5))
plt1=sns.boxplot(dataset['TV'],ax=axs[0])
plt2=sns.boxplot(dataset['Newspaper'],ax=axs[1])
plt3=sns.boxplot(dataset['Radio'],ax=axs[2])
plt.tight_layout()
sns.displot(dataset['Sales']);
sns.pairplot(dataset,x_vars=['TV','Radio','Newspaper'],y_vars='Sales',height=4,aspect=1,kind='scatter')
plt.show()
sns.heatmap(dataset.corr(),annot=True)
plt.show()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
x=dataset[['TV']]
y=dataset['Sales']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=100)
slr=LinearRegression()
slr.fit(x_train,y_train)
print('Intercept:',slr.intercept_)
print('Coefficient:',slr.coef_)
print('Regression Equation:Sales=6.948+0.054*TV')
plt.scatter(x_train,y_train)
plt.plot(x_train,6.948+0.054*x_train,'r')
plt.show()
y_pred_slr=slr.predict(x_test)
x_pred_slr=slr.predict(x_train)
print("Prediction for test set:{}".format(y_pred_slr))
slr_diff=pd.DataFrame({'Actual value':y_test ,'Predicted value':y_pred_slr})
slr_diff
slr.predict([[56]])
from sklearn.metrics import accuracy_score
print('R squared value of the model:{:.2f}'.format(slr.score(x,y)*100))
