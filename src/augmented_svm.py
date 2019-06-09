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
import data_augmentation

seed = 8
np.random.seed(seed)

def feature_normalize(dataset):
    return (dataset - np.mean(dataset, axis=0))/np.std(dataset, axis=0)
	
df1 = pd.read_excel('data/female_session_1.xlsx',  header=None)
input1 = df1.as_matrix()

df2 = pd.read_excel('data/female_session_2.xlsx',  header=None)
input2 = df2.as_matrix()

df3 = pd.read_excel('data/male_session_1.xlsx',  header=None)
input3 = df3.as_matrix()

df4 = pd.read_excel('data/male_session_2.xlsx',  header=None)
input4 = df4.as_matrix()

Y1 = np.ones((144,1), np.float32)
for i in range(0,Y1.shape[0],48):
    if (i == 0):
        Y1[0:48] = Y1[0:48]*0
    if (i == 0):
        Y1[96:] = Y1[96:]*2
		
Y2 = np.ones((144,1), np.float32)
for i in range(0,Y2.shape[0],48):
    if (i == 0):
        Y2[0:48] = Y2[0:48]*0
    if (i == 0):
        Y2[96:] = Y2[96:]*2

Y3 = np.ones((144,1), np.float32)
for i in range(0,Y3.shape[0],48):
    if (i == 0):
        Y3[0:48] = Y3[0:48]*0
    if (i == 0):
        Y3[96:] = Y3[96:]*2
		
Y4 = np.ones((144,1), np.float32)
for i in range(0,Y4.shape[0],48):
    if (i == 0):
        Y4[0:48] = Y4[0:48]*0
    if (i == 0):
        Y4[96:] = Y4[96:]*2

X_aug_1, Y_aug_1 = data_augmentation.get_augmented_input_1()
X_aug_2, Y_aug_2 = data_augmentation.get_augmented_input_2()
X_aug_3, Y_aug_3 = data_augmentation.get_augmented_input_3()
X_aug_4, Y_aug_4 = data_augmentation.get_augmented_input_4()

Y_o = np.vstack([Y1, Y2, Y3, Y4])
Y = np.vstack([Y_o, Y_aug_1, Y_aug_2, Y_aug_3, Y_aug_4]).reshape((2832))

X_input_o = np.vstack([input1, input2, input3, input4])
X_input = np.vstack([X_input_o, X_aug_1, X_aug_2, X_aug_3, X_aug_4])

X_norm = feature_normalize(X_input).reshape((2832, 25))
X = X_norm

class_names = ['eye','man','hand']

x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=4)
x_test, x_dev, y_test, y_dev = train_test_split(x_val, y_val, test_size=0.5, random_state=4) 

svm_model_linear = SVC(kernel = 'rbf', C = 1).fit(x_train, y_train) 

svm_predictions = svm_model_linear.predict(x_test) 
  
accuracy = svm_model_linear.score(x_test, y_test) 
  
cm = confusion_matrix(y_test, svm_predictions)

matrix_plot.plot_confusion_matrix(y_test, svm_predictions, classes=class_names, normalize=True,
                      title='SVM Confusion Matrix on Augmented Data')
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

accuracy = svm_model_linear.score(x_train, y_train) 
print("train accuracy: ", accuracy*100)
print("============================")