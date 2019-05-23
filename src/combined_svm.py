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
	
df1 = pd.read_excel('data/all_data_2.xlsx',  header=None)
input1 = df1.as_matrix()

df2 = pd.read_excel('data/all_data.xlsx',  header=None)
input2 = df2.as_matrix()


Y1 = np.ones((150,1), np.float32)
for i in range(0,Y1.shape[0],50):
    if (i == 0):
        Y1[0:50] = Y1[0:50]*0
    if (i == 0):
        Y1[100:] = Y1[100:]*2
		
Y2 = np.ones((144,1), np.float32)
for i in range(0,Y2.shape[0],48):
    if (i == 0):
        Y2[0:48] = Y2[0:48]*2
    if (i == 0):
        Y2[96:] = Y2[96:]*0

Y = np.vstack([Y1, Y2]).reshape((294))
print(Y)
X_input = np.vstack([input1, input2])

X = X_input.reshape((294,25))
#X_norm = feature_normalize(X_input).reshape((294,25))
#X = X_norm
'''

Y1 = np.ones((75,1), np.float32)
for i in range(0,Y1.shape[0],25):
    if (i == 0):
        Y1[0:25] = Y1[0:25]*0
    if (i == 0):
        Y1[50:] = Y1[50:]*2
		
Y2 = np.ones((72,1), np.float32)
for i in range(0,Y2.shape[0],24):
    if (i == 0):
        Y2[0:24] = Y2[0:24]*2
    if (i == 0):
        Y2[48:] = Y2[48:]*0

Y = np.vstack([Y1, Y2]).reshape((147))
print(Y)
X_input = np.vstack([input1, input2])

X = X_input.reshape((147,50))
#X_norm = feature_normalize(X_input).reshape((294,25))
#X = X_norm
'''

class_names = ['man','hand','eye']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3) 

svm_model_linear = SVC(kernel = 'rbf', C = 1).fit(X_train, y_train) 

svm_predictions = svm_model_linear.predict(X_test) 
  
accuracy = svm_model_linear.score(X_test, y_test) 
  
cm = confusion_matrix(y_test, svm_predictions)


matrix_plot.plot_confusion_matrix(y_test, svm_predictions, classes=class_names,
                      title='Confusion matrix, without normalization')


matrix_plot.plot_confusion_matrix(y_test, svm_predictions, classes=class_names, normalize=True,
                      title='confusion matrix of svm model generated from combined data of session 1 and session 2')
plt.show()

print("Results: ", svm_predictions)
print("True Values: ", y_test)

print("accuracy: ", accuracy*100)