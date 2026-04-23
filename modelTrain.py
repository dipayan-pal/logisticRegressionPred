import pandas as pd 
import numpy as np

dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)    

# ## Predicting a single input 
# result = classifier.predict(sc.transform([[45, 80000]]))
# print(result[0])

## Predicting the Test set results
y_pred = classifier.predict(X_test)
# print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# ## Getting the confusion matrix and accuracy score
# from sklearn.metrics import confusion_matrix, accuracy_score
# cm = confusion_matrix(y_test, y_pred)
# print(cm)
# accuracy = accuracy_score(y_test, y_pred)
# print(accuracy)

## Saving the model
import joblib
joblib.dump(classifier, 'model.pkl')

## Saving the scaler
joblib.dump(sc, 'scaler.pkl')