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

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

def generate_word_one():
    features = np.zeros(81)
    features_n = np.zeros((10,8))
    for i in range(0, 10):
        d = randint(50,100)
        features[i*8] = d
        features_n[i][0] = d
        
        t = randint(350, 400)
        features[i*8 + 1] = t
        features_n[i][1] = t
        
        lA = randint(750, 800)
        features[i*8 + 2] = lA
        features_n[i][2] = lA
        
        hA = randint(1000, 1099)
        features[i*8 + 3] = hA
        features_n[i][3] = hA
        
        lB = randint(1300, 1490)
        features[i*8 + 4] = lB
        features_n[i][4] = lB
        
        hB = randint(1800, 2000)
        features[i*8 + 5] = hB
        features_n[i][5] = hB
        
        lG = randint(3100, 3300)
        features[i*8 + 6] = lG
        features_n[i][6] = lG
        
        hG = randint(4100, 4500)
        features[i*8 + 7] = hG
        features_n[i][7] = hG

    f = features_n/np.linalg.norm(features_n, ord=1, axis=0, keepdims=True)
    features[80] = 1
    return features, np.append(f.reshape(80), 1)
	
def generate_word_two():
    features = np.zeros(81)
    features_n = np.zeros((10,8))
    for i in range(0, 10):
        d = randint(80,200)
        features[i*8] = d
        features_n[i][0] = d

        t = randint(330, 550)
        features[i*8 + 1] = t
        features_n[i][1] = t
		
        lA = randint(780, 900)
        features[i*8 + 2] = lA
        features_n[i][2] = lA
		
        hA = randint(1080, 1130)
        features[i*8 + 3] = hA
        features_n[i][3] = hA
		
        lB = randint(1450, 1600)
        features[i*8 + 4] = lB
        features_n[i][4] = lB
		
        hB = randint(1900, 2400)
        features[i*8 + 5] = hB
        features_n[i][5] = hB
		
        lG = randint(3200, 3805)
        features[i*8 + 6] = lG
        features_n[i][6] = lG
		
        hG = randint(4400, 4700)
        features[i*8 + 7] = hG
        features_n[i][7] = hG

    f = features_n/np.linalg.norm(features_n, ord=1, axis=0, keepdims=True)
    features[80] = 2
    return features, np.append(f.reshape(80), 2)
	
def generate_word_three():
    features = np.zeros(81)
    features_n = np.zeros((10,8))
    for i in range(0, 10):
        d = randint(180,275)
        features[i*8] = d
        features_n[i][0] = d
		
        t = randint(480, 675)
        features[i*8 + 1] = t
        features_n[i][1] = t
		
        lA = randint(850, 925)
        features[i*8 + 2] = lA
        features_n[i][2] = lA
		
        hA = randint(1100, 1175)
        features[i*8 + 3] = hA
        features_n[i][3] = hA
		
        lB = randint(1550, 1675)
        features[i*8 + 4] = lB
        features_n[i][4] = lB
		
        hB = randint(2200, 2975)
        features[i*8 + 5] = hB
        features_n[i][5] = hB
		
        lG = randint(3500, 3975)
        features[i*8 + 6] = lG
        features_n[i][6] = lG
		
        hG = randint(4600, 4975)
        features[i*8 + 7] = hG
        features_n[i][7] = hG

    f = features_n/np.linalg.norm(features_n, ord=1, axis=0, keepdims=True)		
    features[80] = 3
    return features, np.append(f.reshape(80), 3)
	
smaple_count = 10
word1 = []
word2 = []
word3 = []

for i in range(0, smaple_count):
    f1_1, f1_2 = generate_word_one()
    f2_1, f2_2 = generate_word_two()
    f3_1, f3_2 = generate_word_three()
    #print(f1_1.shape, f1_2.shape)
    #print(f1_1, f1_2)
    word1.append(f1_1)
    word2.append(f2_1)
    word3.append(f3_1)

word_1 = np.array(word1)
word_2 = np.array(word2)
word_3 = np.array(word3)

feature_set = np.vstack([word_1, word_2, word_3]) 

labels = np.array([0]*10 + [1]*10 + [2]*10)

one_hot_labels = np.zeros((30, 3))

for i in range(30):  
    one_hot_labels[i, labels[i]] = 1
	
#X_b = feature_set[:,0:80].astype(float)
X = feature_set[:,0:80].astype(float)
Y = feature_set[:,80]

#word_1_r = word_1.reshape((100, 8))
#word_2_r = word_2.reshape((100, 8))
#word_3_r = word_3.reshape((100, 8))
#print("word_1", word_1.shape)
#print("word_2", word_2.shape)
#print("word_3", word_3.shape)

#print("word_1_r", word_1_r.shape)
#print("word_2_r", word_2_r.shape)
#print("word_3_r", word_3_r.shape)

#print("word_1_r", word_1_r)
#print("word_2_r", word_2_r)
#print("word_3_r", word_3_r)
#word_1 = np.random.randn(10, 80) + np.array([0, -3, 0, -3, 0, -3, 0, -3])  
#word_2 = np.random.randn(10, 80) + np.array([3, 3, 3, 3, 3, 3, 3, 3])  
#word_3 = np.random.randn(10, 80) + np.array([-3, 3, -3, 3, -3, 3, -3, 3])

#word_1_n = word_1_r/np.linalg.norm(word_1_r, ord=2, axis=0, keepdims=True)
#word_2_n = word_2_r/np.linalg.norm(word_2_r, ord=2, axis=0, keepdims=True)
#word_3_n = word_3_r/np.linalg.norm(word_3_r, ord=2, axis=0, keepdims=True)

#print("word_1_n", word_1_n.shape)
#print("word_2_n", word_2_n.shape)
#print("word_3_n", word_3_n.shape)

#print("word_n_r", word_1_n)
#print("word_n_r", word_2_n)
#print("word_n_3", word_3_n)

#word_1_b = word_1_n.reshape((10, 80))
#word_2_b = word_2_n.reshape((10, 80))
#word_3_b = word_3_n.reshape((10, 80))

#X = X_b.reshape((100, 8))
#X = X_b/np.linalg.norm(X_b, ord=2, axis=0, keepdims=True)
#X = word_1_n.reshape((10, 80))

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(8, input_dim=80, activation='relu'))
	model.add(Dense(20, input_dim=8, activation='relu'))
	#model.add(Dense(30, input_dim=20, activation='relu'))
	#model.add(Dense(10, input_dim=30, activation='relu'))
	model.add(Dense(3, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

#X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=seed)	
#baseline_model().fit(X_train, y_train, validation_data=(X_test,y_test), epochs=150, batch_size=1)

estimator = KerasClassifier(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))