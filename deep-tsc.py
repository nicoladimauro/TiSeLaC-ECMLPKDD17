import numpy as np
np.random.seed(1379)

from keras.utils import plot_model

from sklearn.neighbors import BallTree
from sklearn import preprocessing
from sklearn import svm
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.linear_model import LogisticRegression as LR

from collections import Counter

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Merge, Activation, Conv1D, MaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
import keras
from keras import backend as K
from keras.layers.advanced_activations import PReLU, ELU
from keras.optimizers import SGD
from keras import regularizers


import my_callbacks
from models import build_conv_models


### load data
train_coord = np.loadtxt("data/coord_training.txt", delimiter=',')
X_train_ = np.loadtxt('data/training.txt.gz', delimiter=",")
y_train = np.loadtxt("data/training_class.txt")

X_test_ = np.loadtxt('data/test.txt', delimiter=",")
test_coord = np.loadtxt("data/coord_test.txt", delimiter=',')
y_test = np.loadtxt("data/test.cl")

num_classes = 9

### build the complete spatial feature descriptor
kdt = BallTree(train_coord, leaf_size=30, metric='euclidean')

train_neigh = np.empty(shape=(1, 0))
test_neigh = np.empty(shape=(1, 0))

e_X_train = X_train_.reshape(X_train_.shape[0], 23, 10)
e_X_train = np.transpose(e_X_train, (0, 2, 1))
e_X_test = X_test_.reshape(X_test_.shape[0], 23, 10)
e_X_test = np.transpose(e_X_test, (0, 2, 1))

for radius in [1, 3, 5, 7, 9, 11, 13, 15, 17]:
    print('computing ball for radius {}'.format(radius))

    train_neighbors = kdt.query_radius(train_coord, r=radius)
    test_neighbors = kdt.query_radius(test_coord, r=radius)

    neig = []
    for i in range(X_train_.shape[0]):
        mask_ = train_neighbors[i]
        index = np.argwhere(mask_ == i)
        mask = np.delete(mask_, index)

        classes = y_train[mask]
        unique, counts = np.unique(classes, return_counts=True)
        N = [0] * num_classes
        for j in range(len(unique)):
            N[int(unique[j] - 1)] = counts[j]

        if len(mask) < 1:
            N.extend([0., 0., 0., 0., 0., 0.])
        else:
            N_i = e_X_train[mask]

            N.append(np.mean(N_i[:, 7]) + 0.001)
            N.append(np.std(N_i[:, 7]) + 0.001)
            N.append(np.mean(N_i[:, 8]) + 0.001)
            N.append(np.std(N_i[:, 8]) + 0.001)
            N.append(np.mean(N_i[:, 9]) + 0.001)
            N.append(np.std(N_i[:, 9]) + 0.001)

        neig.append(N)

    if (radius == 1):
        train_neigh = np.array(neig)
    else:
        train_neigh = np.concatenate((train_neigh, np.array(neig)), axis=1)

    neig = []
    for i in range(X_test_.shape[0]):
        mask_ = test_neighbors[i]
        index = np.argwhere(mask_ == i)
        mask = np.delete(mask_, index)

        classes = y_train[mask]
        unique, counts = np.unique(classes, return_counts=True)
        N = [0] * num_classes
        for j in range(len(unique)):
            N[int(unique[j] - 1)] = counts[j]

        if len(mask) < 1:
            N.extend([0., 0., 0., 0., 0., 0.])
        else:
            N_i = e_X_train[mask]

            N.append(np.mean(N_i[:, 7]) + 0.001)
            N.append(np.std(N_i[:, 7]) + 0.001)
            N.append(np.mean(N_i[:, 8]) + 0.001)
            N.append(np.std(N_i[:, 8]) + 0.001)
            N.append(np.mean(N_i[:, 9]) + 0.001)
            N.append(np.std(N_i[:, 9]) + 0.001)

        neig.append(N)

    if (radius == 1):
        test_neigh = np.array(neig)
    else:
        test_neigh = np.concatenate((test_neigh, np.array(neig)), axis=1)

train_neigh = np.concatenate((train_neigh, train_coord), axis=1)
test_neigh = np.concatenate((test_neigh, test_coord), axis=1)

X_train_aggr = np.reshape(X_train_, (-1, 10))
X_test_aggr = np.reshape(X_test_, (-1, 10))


### scaling 
scaler = preprocessing.StandardScaler().fit(X_train_aggr)
X_train = scaler.transform(X_train_aggr)
X_test = scaler.transform(X_test_aggr)

X_train = np.reshape(X_train, (-1, 230))
X_test = np.reshape(X_test, (-1, 230))

X_train_t = np.reshape(X_train, (X_train.shape[0], 230, 1))
X_test_t = np.reshape(X_test, (X_test.shape[0], 230, 1))

X_train = X_train.reshape(X_train.shape[0], 23, 10)
X_test = X_test.reshape(X_test.shape[0], 23, 10)

input_shape = (23, 10)


scaler = preprocessing.StandardScaler().fit(train_neigh)
X_train_neigh = scaler.transform(train_neigh)
X_test_neigh = scaler.transform(test_neigh)

lb = preprocessing.LabelBinarizer()
lb.fit(y_train)

y_train = lb.transform(y_train)
y_test = lb.transform(y_test)


### spectral model
spectral_model = Sequential()
spectral_model.add(
    Conv1D(filters=32, kernel_size=3, input_shape=input_shape, activation='relu', padding='same'))
spectral_model.add(Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'))
spectral_model.add(Flatten())

ch_models = []
ch_maps = []
input_list = []
n_models = 10
for t in range(n_models):
    map_ch = [i * 10 + t for i in range(23)]
    ch_maps.append(map_ch)

TRAIN = [X_train] + [X_train_t[:, ch_maps[i]] for i in range(n_models)] + [X_train_neigh]
TEST = [X_test] + [X_test_t[:, ch_maps[i]] for i in range(n_models)] + [X_test_neigh]

spectral_model2, layers = build_conv_models(n_models=10, input_shape=(23, 1))


relational_model = Sequential()
relational_model.add(Dense(128, input_shape=(X_train_neigh.shape[1],), activation='relu'))
relational_model.add(Dropout(0.3))
relational_model.add(Dense(64,  activation='relu'))
relational_model.add(Dropout(0.3))

complete_model = Sequential()
complete_model.add(Merge([spectral_model, spectral_model2, relational_model], mode='concat'))
complete_model.add(Dense(9, activation='softmax'))

complete_model.compile(loss='categorical_crossentropy',
                       optimizer=keras.optimizers.Adam(),
                       metrics=['accuracy'])

plot_model(complete_model, to_file='model.png', show_shapes=True, show_layer_names=False)

complete_model.fit(TRAIN, y_train, batch_size=32,
                   epochs=7, verbose=1)


print("Train")

y_pred_proba = complete_model.predict(TRAIN)
y_pred = np.argmax(y_pred_proba, axis=1) + 1

y_train_ = np.argmax(y_train, axis=1) + 1

#print(confusion_matrix(y_train_, y_pred))
#print(classification_report(y_train_, y_pred))
print("f1 score micro: ", f1_score(y_train_, y_pred, average='micro'))
print("f1 score macro: ", f1_score(y_train_, y_pred, average='macro'))
print("f1 score weighted: ", f1_score(y_train_, y_pred, average='weighted'))

lr = LR()
lr.fit(y_pred_proba, y_train_)
y_pred = lr.predict(y_pred_proba)
print("Train LR")
#print(confusion_matrix(y_train_, y_pred))
#print(classification_report(y_train_, y_pred))
print("f1 score micro: ", f1_score(y_train_, y_pred, average='micro'))
print("f1 score macro: ", f1_score(y_train_, y_pred, average='macro'))
print("f1 score weighted: ", f1_score(y_train_, y_pred, average='weighted'))

print("Test")

y_pred_proba = complete_model.predict(TEST)
y_pred = np.argmax(y_pred_proba, axis=1) + 1

np.savetxt("baML.txt", y_pred, fmt="%.1f", delimiter="\n")

y_test_ = np.argmax(y_test, axis=1) + 1

#print(confusion_matrix(y_test_, y_pred))
#print(classification_report(y_test_, y_pred))
print("f1 score micro: ", f1_score(y_test_, y_pred, average='micro'))
print("f1 score macro: ", f1_score(y_test_, y_pred, average='macro'))
print("f1 score weighted: ", f1_score(y_test_, y_pred, average='weighted'))

lr = LR()
lr.fit(y_pred_proba, y_test_)
y_pred = lr.predict(y_pred_proba)
print("Test LR")
#print(confusion_matrix(y_test_, y_pred))
#print(classification_report(y_test_, y_pred))
print("f1 score micro: ", f1_score(y_test_, y_pred, average='micro'))
print("f1 score macro: ", f1_score(y_test_, y_pred, average='macro'))
print("f1 score weighted: ", f1_score(y_test_, y_pred, average='weighted'))
