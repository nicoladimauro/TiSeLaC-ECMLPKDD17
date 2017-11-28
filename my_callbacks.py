import keras
from sklearn.metrics import f1_score
import numpy as np

class f1_score_cb(keras.callbacks.Callback):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def on_train_begin(self, logs={}):
        self.f1_scores = []

    def on_train_end(self, logs={}):
        print(self.f1_scores)
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred_proba = self.model.predict(self.X)
        y_pred = np.argmax(y_pred_proba, axis=1)+1 
        y_true = np.argmax(self.y, axis=1)+1

        score = f1_score(y_true, y_pred, average='weighted')

        self.f1_scores.append(score)
        print(" val f1_score: ", score)
        return


    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return
