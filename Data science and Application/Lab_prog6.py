import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('iris.csv')
dataset.describe()
dataset.info()

X = dataset.iloc[:,[0,1,2,3]].values
y = dataset.iloc[:,4].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0, solver='lbfgs', multi_class='auto')
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
probs_y = classifier.predict_proba(X_test)
probs_y = np.round(probs_y, 2)

res = "{:<10}|{:<10}|{:<10}|{:<13}|{:<10}".format("y_test", "y_pred", "Setosa(%)", "Versicolor(%)", "Virginica(%)")
res += "\n" + "-" * 65 + "\n"

for x, y, a, b, c in zip(y_test, y_pred, probs_y[:, 0], probs_y[:, 1], probs_y[:, 2]):
    res += "{:<10}|{:<10}|{:<10}|{:<13}|{:<10}\n".format(x, y, a, b, c)

print(res)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

ax = plt.axes()
df_cm = cm
sns.heatmap(df_cm, annot=True, annot_kws={"size": 30}, fmt='d', cmap="Blues", ax=ax)
ax.set_title('Confusion Matrix')
plt.show()
