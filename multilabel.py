from __future__ import print_function
import numpy as np
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, Flatten, Reshape
from pandas import read_csv
#from load_embedding import load_embeddings
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import one_hot
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer

print('Loading data...')
path_train_x = "Xtrain"
path_test_x = "Xtest"
path_dev_x = "Xvalid"
path_train_y = "Ytrain"
path_test_y = "Ytest"
path_dev_y = "Yvalid"

#parameters
epochs = 100
batch_size = 32
inputdim = 50
num_classes = 3553

def load_file(filepath):
        dataframe = read_csv(filepath, header=None, delim_whitespace=True)
        return dataframe.values

def loadlabelfiles(filename):
        with open(filename,'r') as f:
                c = f.readlines()
        data = [x.strip().split() for x in c] 
        return data
        
x_train = load_file(path_train_x)
x_test = load_file(path_test_x)
y_train=loadlabelfiles(path_train_y)
y_test = loadlabelfiles(path_test_y)
y_dev = loadlabelfiles(path_dev_y)

print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')
print(len(x_dev), 'val sequences')


x_train = x_train.astype('float32')

mlb = MultiLabelBinarizer()
y_train_hot = mlb.fit_transform(y_train)
y_test_hot = mlb.fit_transform(y_test)
y_dev_hot = mlb.fit_transform(y_dev)

print('Build model...')
model_m = Sequential()
model_m.add(Dense(256, input_dim=inputdim, kernel_initializer='he_uniform', activation='relu'))
model_m.add(Dense(128, activation='relu'))
model_m.add(Dense(num_classes, activation='sigmoid'))

model_m.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model_m.summary())

model_m.fit(x_train, y_train_hot, 
          validation_data=(x_dev, y_dev_hot),
          batch_size = batch_size,
          epochs = epochs)

y_pred = model_m.predict(x_test)
y_pred = y_pred.round()
y_pred_hot = mlb.fit_transform(y_pred)

loss = model_m.evaluate(x_test, y_test_hot)[0]
accuracy = model_m.evaluate(x_test, y_test_hot)[1]
print('loss', loss)
print('accuracy', accuracy)

print('-----------y test hot--------')
print(y_test_hot)
print(y_test_hot.shape)

print('-----------y pred ----------')
print(y_pred)
print(y_pred.shape)

f1_macro = f1_score(y_test_hot, y_pred, average='macro')
f1_micro = f1_score(y_test_hot, y_pred, average='micro')
f1_normal = f1_score(y_test_hot, y_pred, average=None)

print('loss', loss)
print('accuracy', accuracy)
print('f1_macro', f1_macro)
print('f1_micro', f1_micro)
print('f1_normal', f1_normal)

model_m.save("model.h5")
print("Saved model to disk")

f = open("results.txt", "w")
for i in range(len(x_test)):
    f.write(str(y_test[i]))
    f.write("\t")
    f.write(str(y_pred[i]))
    f.write("\n")
f.close()