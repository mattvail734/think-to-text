from sklearn import datasets 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

import pandas as pd 
import numpy as np
import matrix_plot

import matplotlib.pyplot as plt

from sklearn.svm import SVC 

seed = 8
np.random.seed(seed)

def feature_normalize(dataset):
    return (dataset - np.mean(dataset, axis=0))/np.std(dataset, axis=0)
	
df = pd.read_excel('data/all_data_2.xlsx',  header=None)
input = df.as_matrix()

#X=input.reshape((150,25))
X_norm = feature_normalize(input).reshape((150,25))
#X_norm = feature_normalize(input.reshape((150,25)))
X = X_norm

Y = np.ones((150), np.float32)
for i in range(0,Y.shape[0],50):
    if (i == 0):
        Y[0:50] = Y[0:50]*0
    if (i == 0):
        Y[100:] = Y[100:]*2
'''
#X=input.reshape((75,50))
X_norm = feature_normalize(input).reshape((75,50))
X = X_norm

Y = np.ones((75), np.float32)
for i in range(0,Y.shape[0],25):
    if (i == 0):
        Y[0:25] = Y[0:25]*0
    if (i == 0):
        Y[50:] = Y[50:]*2
'''

class_names = [0,1,2]

X_train, X_test, y_train, y_test = train_test_split(X, Y) 

svm_model_linear = SVC(kernel = 'rbf', C = 1).fit(X_train, y_train) 

svm_predictions = svm_model_linear.predict(X_test) 
  
accuracy = svm_model_linear.score(X_test, y_test) 
  
cm = confusion_matrix(y_test, svm_predictions)

# Plot non-normalized confusion matrix
matrix_plot.plot_confusion_matrix(y_test, svm_predictions, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
matrix_plot.plot_confusion_matrix(y_test, svm_predictions, classes=class_names, normalize=True,
                      title='confusion matrix of svm model generated from session 1 data')
plt.show()

print("Train Count: ", y_train.shape)
print("============================")
print("Test Count: ", y_test.shape)
print("============================")
print("Results: ", svm_predictions)
print("============================")
print("True Values: ", y_test)
print("============================")
print("accuracy: ", accuracy*100)
print("============================")

df2 = pd.read_excel('data/all_data.xlsx',  header=None)
input2 = df2.as_matrix()


#X2=input2.reshape((144,25))
X_norm2 = feature_normalize(input2).reshape((144,25))
#X_norm2 = feature_normalize(input2.reshape((144,25)))
X2 = X_norm2

Y2 = np.ones((144), np.float32)
for i in range(0,Y2.shape[0],48):
    if (i == 0):
        Y2[0:48] = Y2[0:48]*2
    if (i == 0):
        Y2[96:] = Y2[96:]*0
'''

#X2=input2.reshape((72,50))
X_norm2 = feature_normalize(input2).reshape((72,50))
X2 = X_norm2

Y2 = np.ones((72), np.float32)
for i in range(0,Y2.shape[0],24):
    if (i == 0):
        Y2[0:24] = Y2[0:24]*2
    if (i == 0):
        Y2[48:] = Y2[48:]*0
'''		
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, Y2, test_size=0.2)

svm_predictions2 = svm_model_linear.predict(X_test2)

cm2 = confusion_matrix(y_test2, svm_predictions2)

# Plot non-normalized confusion matrix
matrix_plot.plot_confusion_matrix(y_test2, svm_predictions2, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
matrix_plot.plot_confusion_matrix(y_test2, svm_predictions2, classes=class_names, normalize=True,
                      title='model from session 1 to classify data from session 2')
plt.show()

accuracy2 = svm_model_linear.score(X_test2, y_test2)
print("Test Count: ", y_test2.shape)
print("============================")
print("accuracy: ", accuracy2*100)