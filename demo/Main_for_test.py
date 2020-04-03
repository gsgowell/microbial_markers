import scipy.io as sio
import numpy as np
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.core import Dense, Activation
from keras.models import Sequential, save_model, load_model
from keras.utils import to_categorical
from sklearn.metrics import roc_auc_score, roc_curve,auc, confusion_matrix


X = np.load('D1_minus_RF_MarkerSpecies_X.npy')     # D1_minus_RF_MarkerSpecies_X.npy; D1_minus_LEfSe_MarkerSpecies_X.npy; D1_minus_MarkerGenes_X
Y = np.load('D1_minus_lables_Y.npy')

X_test = np.load('D2_plus_RF_MarkerSpecies_X.npy')  # D2_plus_RF_MarkerSpecies_X.npy; D2_plus_LEfSe_MarkerSpecies_X.npy; D2_plus_MarkerGenes_X
Y_test = np.load('D2_plus_labels_Y.npy')


X = X[:, 0:40]  # species: 40; gene: 90
X_test = X_test[:, 0:40] # species: 40; gene: 90

X = np.array(X)
Y = np.array(Y)
X_test = np.array(X_test)
Y_test = np.array(Y_test)
YY = Y_test

batch_size = 3         # 2,3, 4, 5
num_classes = 2
epochs = 50             # 35, 40, 45, 50, ..., 100, 110, 120,..., 200,...250
(nsize, nf) = X.shape

Y_train = to_categorical(Y, num_classes)
Y_test = to_categorical(Y_test, num_classes)

# normalize the train dataset

X_train= (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
X_test = (X_test - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))


def get_F1(precision, recall):
    """F1-score"""
    f1 = 2 * ((precision * recall) / (precision + recall))
    return f1

'''
model = Sequential()
model.add(Dense(input_dim=nf, units=18))
model.add(Activation('relu'))
model.add(Dense(9))
model.add(Activation('relu'))
model.add(Dense(num_classes))
model.add(Activation('softmax'))                   #relu

model.compile(loss='categorical_crossentropy',
               optimizer=Adam(),
               metrics=['acc'])    # categorical_crossentropy  mse
               
model.fit(X_train, Y_train,batch_size=batch_size,epochs=epochs,verbose=1)  # validation_data=(x_test, y_test), verbose = 1
'''

# load model
model = load_model('RF_MarkerSpecies_D1_minus_trained.h5')  # RF_MarkerGenes_D1_minus_trained.h5; LEfSe_MarkerSpecies_D1_minus_trained.h5

scores = model.evaluate(X_test, Y_test)
y_pred = model.predict_classes(X_test)
print('acc:' + str(scores[1]))
cm = confusion_matrix(YY, y_pred)
tp = cm[1][1]
fp = cm[0][1]
tn = cm[0][0]
fn = cm[1][0]

precision = tp/(tp + fp)
recall = tp/(tp + fn)
f1 = get_F1(precision, recall)
print('precision:' + str(precision))
print('recall:' + str(recall))
print('f1:' + str(f1))

