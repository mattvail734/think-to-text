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

class_names = ['man','hand','eye']
		
X_train, X_test, y_train, y_test = train_test_split(X, Y) 

dtree_model = DecisionTreeClassifier(max_depth = 4).fit(X_train, y_train) 
dtree_predictions = dtree_model.predict(X_test)

cm = confusion_matrix(y_test, dtree_predictions)

# Plot normalized confusion matrix
matrix_plot.plot_confusion_matrix(y_test, dtree_predictions, classes=class_names, normalize=True,
                      title='Decision Tree From Session 1 Data')
plt.show()

count = 0
for i in range (y_test.shape[0]):
    if y_test[i] == dtree_predictions[i]:
        #print(dtree_predictions[i])
        #print(y_test[i])
        count += 1
        #print(count)
	
#print(count)
#print(y_test.shape[0])	

print("Train Count: ", y_train.shape)
print("============================")
print("Test Count: ", y_test.shape)
print("============================")
print("Results: ", dtree_predictions)
print("============================")
print("True Values: ", y_test)
print("============================")
print("accuracy: ", count*100/y_test.shape[0])
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
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, Y2, test_size=0.9)
		
dtree_predictions2 = dtree_model.predict(X_test2)

cm2 = confusion_matrix(y_test2, dtree_predictions2)

# Plot normalized confusion matrix
matrix_plot.plot_confusion_matrix(y_test2, dtree_predictions2, classes=class_names, normalize=True,
                      title='Decision Tree From session 1 Used On Session 2')
plt.show()

count2 = 0
for i in range (y_test2.shape[0]):
    if y_test2[i] == dtree_predictions2[i]:
        #print(dtree_predictions[i])
        #print(y_test[i])
        count2 += 1
        #print(count)
	
#print(count)
#print(y_test.shape[0])	
print("Results: ", dtree_predictions2)
print("============================")
print("True Values: ", y_test2)
print("============================")
print("accuracy: ", count*100/y_test2.shape[0])
print("============================")