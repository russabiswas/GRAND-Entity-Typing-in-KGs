from __future__ import print_function
import numpy as np
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, Flatten,Reshape
from keras.layers.convolutional import MaxPooling1D
from pandas import read_csv
#from load_embedding import load_embeddings
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import one_hot
from sklearn.metrics import f1_score


def load_file(filepath):
        dataframe = read_csv(filepath, header=None, delim_whitespace=True)
        return dataframe.values

def model(path_train_x,path_test_x,path_dev_x,path_train_y,path_test_y,path_dev_y, num_classes, level):
        batch_size = 32
        epochs = 25
        input_dim = 600
        
        x_train = load_file(path_train_x)
        x_test = load_file(path_test_x)
        x_dev = load_file(path_dev_x)
        y_train = np.loadtxt(fname = path_train_y)
        y_test = np.loadtxt(fname = path_test_y)
        y_dev = load_file(path_dev_y)
        
        print(len(x_train), 'train sequences')
        print(len(x_test), 'test sequences')
        print(len(x_dev), 'val sequences')

        x_train = x_train.astype('float32')
        y_train = y_train.astype('float32')

        y_train_hot = np_utils.to_categorical(y_train)
        print('New y_train shape: ', y_train_hot.shape) 
        y_test_hot = np_utils.to_categorical(y_test)
        print('New y_test shape: ', y_test_hot.shape) 
        y_dev_hot = np_utils.to_categorical(y_dev)
        print('New y_dev shape: ', y_dev_hot.shape) 

        print('Build model...')
        model_m = Sequential()

        model_m.add(Reshape((input_dim,1), input_shape=(input_dim,)))
        model_m.add(Flatten())
        model_m.add(Dense(256,activation='relu'))
        model_m.add(Dense(128, activation='relu'))
        model_m.add(Dense(num_classes, activation='softmax'))

        model_m.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        print(model_m.summary())

        model_m.fit(x_train, y_train_hot, 
          validation_data=(x_dev, y_dev_hot),
          batch_size = batch_size,
          epochs= epochs)
        y_pred = model_m.predict_classes(x_test)

        loss = model_m.evaluate(x_test, y_test_hot)[0]
        accuracy = model_m.evaluate(x_test, y_test_hot)[1]
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_micro = f1_score(y_test, y_pred, average='micro')
        f1_normal = f1_score(y_test, y_pred, average=None)

        print('loss', loss)
        print('accuracy', accuracy)
        print('f1_macro', f1_macro)
        print('f1_micro', f1_micro)
        print('f1_normal', f1_normal)

        model_m.save(str(level)+"_RDF2Vec_oa.h5")
        print("Saved model to disk")

        f = open(str(level)+"_RDF2Vec_oa.txt", "a")
        for i in range(len(x_test)):
            f.write(str(y_test[i]))    
            f.write("\t")
            f.write(str(y_pred[i]))
            f.write("\n")
        f.close()
        return accuracy, f1_macro, f1_micro


if __name__ == "__main__":

        num_classes_level1 = 5
        num_classes_level2 = 11
        num_classes_level3 = 12
        num_classes_level4 = 17
        acc1, f1ma1, f1mi1 = model("DB1_Xtrain_level1", "DB1_Xtest_level1", "DB1_Xdev_level1",
                                "DB1_Ytrain_level1", "DB1_Ytest_level1", "DB1_Ydev_level1", num_classes_level1, 1)
        acc2, f1ma2, f1mi2 = model("DB1_Xtrain_level2", "DB1_Xtest_level2", "DB1_Xdev_level2", 
                                "DB1_Ytrain_level2", "DB1_Ytest_level2", "DB1_Ydev_level2",num_classes_level2 , 2)
        acc3, f1ma3, f1mi3 = model("DB1_Xtrain_level3", "DB1_Xtest_level3", "DB1_Xdev_level3", 
                                "DB1_Ytrain_level3", "DB1_Ytest_level3", "DB1_Ydev_level3", num_classes_level3, 3)
        acc4, f1ma4, f1mi4 = model("DB1_Xtrain_level4", "DB1_Xtest_level4", "DB1_Xdev_level4", 
                                "DB1_Ytrain_level4", "DB1_Ytest_level4", "DB1_Ydev_level4", num_classes_level4, 4)
        
        
        print('Level 1: Accuracy, Macro-F1, Micro-F1: ',acc1, f1ma1, f1mi1)
        print('Level 2: Accuracy, Macro-F1, Micro-F1: ',acc2, f1ma2, f1mi2)
        print('Level 3: Accuracy, Macro-F1, Micro-F1: ',acc3, f1ma3, f1mi3)
        print('Level 4: Accuracy, Macro-F1, Micro-F1: ',acc3, f1ma4, f1mi4)
        acc  = (acc1+acc2+acc3+acc4)/4
        f1ma = (f1ma1 + f1ma2 + f1ma3 + f1ma4)/4
        f1mi = (f1mi1 + f1mi2 + f1mi3 + f1mi4)/4
        print('Overall Results')
        print(acc, f1ma, f1mi)




