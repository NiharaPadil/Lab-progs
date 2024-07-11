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
