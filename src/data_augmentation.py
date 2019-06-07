import pandas as pd 
import numpy as np

import matplotlib.pyplot as plt
from random import randint

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from scipy.signal import butter, lfilter, iirnotch

from keras.layers import Dense, Dropout, Activation,Lambda,Input,LSTM,GlobalAveragePooling1D
from keras.layers import Conv1D, MaxPooling1D,Flatten,TimeDistributed,Reshape
from keras.utils import np_utils
from keras import losses
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D

df1 = pd.read_excel('data/female_session_1.xlsx',  header=None)
input1 = df1.as_matrix()

df2 = pd.read_excel('data/female_session_2.xlsx',  header=None)
input2 = df2.as_matrix()

df3 = pd.read_excel('data/male_session_1.xlsx',  header=None)
input3 = df3.as_matrix()

df4 = pd.read_excel('data/male_session_2.xlsx',  header=None)
input4 = df4.as_matrix()

Y1 = np.ones((141,1), np.float32)
for i in range(0,Y1.shape[0],48):
    if (i == 0):
        Y1[0:47] = Y1[0:47]*0
    if (i == 0):
        Y1[94:] = Y1[94:]*2
		
Y2 = np.ones((141,1), np.float32)
for i in range(0,Y2.shape[0],48):
    if (i == 0):
        Y2[0:47] = Y2[0:47]*0
    if (i == 0):
        Y2[94:] = Y2[94:]*2

Y3 = np.ones((141,1), np.float32)
for i in range(0,Y3.shape[0],48):
    if (i == 0):
        Y3[0:47] = Y3[0:47]*0
    if (i == 0):
        Y3[94:] = Y3[94:]*2
		
Y4 = np.ones((141,1), np.float32)
for i in range(0,Y4.shape[0],48):
    if (i == 0):
        Y4[0:47] = Y4[0:47]*0
    if (i == 0):
        Y4[94:] = Y4[94:]*2

Y = np.vstack([Y1, Y2, Y3, Y4])

X_input_1 = np.vstack([input1[1:236,:], input1[241:476,:], input1[481:716,:], input2[1:236,:], input2[241:476,:], input2[481:716,:], input3[1:236,:], input3[241:476,:], input3[481:716,:], input4[1:236,:], input4[241:476,:], input4[481:716,:]])
def get_augmented_input_1():
    return X_input_1, Y
	
X_input_2 = np.vstack([input1[2:237,:], input1[242:477,:], input1[482:717,:], input2[2:237,:], input2[242:477,:], input2[482:717,:], input3[2:237,:], input3[242:477,:], input3[482:717,:], input4[2:237,:], input4[242:477,:], input4[482:717,:]])
def get_augmented_input_2():
    return X_input_2, Y
	
X_input_3 = np.vstack([input1[3:238,:], input1[243:478,:], input1[483:718,:], input2[3:238,:], input2[243:478,:], input2[483:718,:], input3[3:238,:], input3[243:478,:], input3[483:718,:], input4[3:238,:], input4[243:478,:], input4[483:718,:]])
def get_augmented_input_3():
    return X_input_3, Y
	
X_input_4 = np.vstack([input1[4:239,:], input1[244:479,:], input1[484:719,:], input2[4:239,:], input2[244:479,:], input2[484:719,:], input3[4:239,:], input3[244:479,:], input3[484:719,:], input4[4:239,:], input4[244:479,:], input4[484:719,:]])
def get_augmented_input_4():
    return X_input_4, Y