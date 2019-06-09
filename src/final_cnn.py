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
import data_augmentation

import matplotlib.pyplot as plt
import matrix_plot

seed = 8
np.random.seed(seed)

def feature_normalize(dataset):
    return (dataset - np.mean(dataset, axis=0))/np.std(dataset, axis=0)
	
df1 = pd.read_excel('data/female_session_1.xlsx',  header=None)
input1 = df1.as_matrix()
input1_n = feature_normalize(input1)

df2 = pd.read_excel('data/female_session_2.xlsx',  header=None)
input2 = df2.as_matrix()
input2_n = feature_normalize(input2)

df3 = pd.read_excel('data/male_session_1.xlsx',  header=None)
input3 = df3.as_matrix()
input3_n = feature_normalize(input3)

df4 = pd.read_excel('data/male_session_2.xlsx',  header=None)
input4 = df4.as_matrix()
input4_n = feature_normalize(input4)

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

		
#X_aug_1, Y_aug_1 = data_augmentation.get_augmented_input_1()
#X_aug_2, Y_aug_2 = data_augmentation.get_augmented_input_2()
#X_aug_3, Y_aug_3 = data_augmentation.get_augmented_input_3()
#X_aug_4, Y_aug_4 = data_augmentation.get_augmented_input_4()

Y = np.vstack([Y1, Y2, Y3, Y4])
#Y = np.vstack([Y_o, Y_aug_1, Y_aug_2, Y_aug_3, Y_aug_4]).reshape((2832))
#print(Y)
X_input = np.vstack([input1, input2, input3, input4])
#print(X_input_o.shape)
#X_input = np.vstack([X_input_o, X_aug_1, X_aug_2, X_aug_3, X_aug_4])
#print(X_input.shape)
#X_input_n = np.vstack([input1_n, input2_n, input3_n, input4_n])

#X = X_input_n.reshape((576,5,5))
#X = X_input.reshape((576,5,5))
X_norm = feature_normalize(X_input).reshape((576,5,5))
X = X_norm

Y_c = np_utils.to_categorical(Y, 3)


# Downsample, shuffle and split (from sklearn.cross_validation)
x_train, x_val, y_train, y_val = train_test_split(X, Y_c, test_size=0.2, random_state=4)
x_test, x_dev, y_test, y_dev = train_test_split(x_val, y_val, test_size=0.5, random_state=4)

# Create the network
model = Sequential()

model.add(Conv1D(16, 3, strides=2, padding='same', activation='relu', input_shape=(5, 5)))
model.add(Dropout(0.2))
#model.add(BatchNormalization())
#model.add(MaxPooling1D(2))

model.add(Conv1D(32, 2, strides=2, padding='same', activation='relu'))
model.add(Dropout(0.1))
#model.add(MaxPooling1D(2))

model.add(Conv1D(64, 2, strides=1, padding='same', activation='relu'))
model.add(Dropout(0.2))
#model.add(MaxPooling1D(2))

model.add(GlobalAveragePooling1D())
#model.add(Dense(64, activation='relu'))
#model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax'))


INIT_LR = 1e-3
EPOCHS = 250

opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
#opt = Adam(lr=INIT_LR)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

# Train and save results for later plotting
history = model.fit(x_train, y_train, batch_size=8, epochs=EPOCHS, validation_data=(x_dev,y_dev))
print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('CNN Accuracy on Original Data')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('CNN Loss on Original Data')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

y_predict = model.predict_classes(x_test)
for i in range(y_predict.shape[0]):
    print("predicted Y: ", y_predict[i], " expected Y: ", np.argmax(y_test[i]))
	
class_names = ['eye','man','hand']
matrix_plot.plot_confusion_matrix(np.argmax(y_test, axis=1), y_predict, classes=class_names, normalize=True,
                      title='CNN Confusion Matrix on Original Data')

plt.show()