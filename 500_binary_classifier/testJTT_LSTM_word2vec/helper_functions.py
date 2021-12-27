import pandas as pd
import numpy
from collections import Counter
import imblearn
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import numpy
from sklearn.metrics import confusion_matrix
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf


###
# IMBALANCED CLASSES MITIGATION
###

def undersample(X, y):
    ''' Undersamples good code samples so we get equally large sets for both classes'''
    
    dataset = pd.DataFrame({"X": list(X), "y": y})
    
    good = dataset[dataset.y == 0].sample(n=dataset.y.value_counts()[1], random_state=0)
    bad  = dataset[dataset.y == 1]
    dataset_balanced = pd.concat([good,bad], axis=0)
    
    return numpy.vstack(dataset_balanced.X), dataset_balanced.y.values
    
    
def oversample_smote(X, y):
    #
    # SMOTE implementation derived from
    # https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/
    #
    
    print ("imbalanced_learn version ", imblearn.__version__)

    # summarize the new class distribution
    counter = Counter(y)
    print("Counter output before SMOTE:", counter)
    
    # transform the dataset
    oversample = SMOTE()
    X, y = oversample.fit_resample(X, y)
    
    # summarize the new class distribution
    counter = Counter(y)
    print("Counter output after SMOTE:", counter)
    
    return X, y


def no_mitigation(X_train, Y_train):
    return X_train, Y_train


##
# MODELS
##

def create_model_MLP(init_mode='glorot_uniform', num_classes=2):
    return keras.Sequential(
        [
            layers.Dense(64,
                         input_dim=100,
                         kernel_initializer=init_mode,
                         activation=tf.nn.relu),
            
            
            layers.Dropout(0.1),
                        
            layers.Dense(64,
                         kernel_initializer=init_mode,
                         activation=tf.nn.relu),
            
            layers.Dense(32,
                         kernel_initializer=init_mode,
                         activation=tf.nn.relu),
            
            
            layers.Dense(num_classes, 
                         kernel_initializer=init_mode, 
                         activation=tf.nn.softmax),
        ]
    )


def create_MLP(init_mode='glorot_uniform', hidden_size=10, num_classes=2):
    return keras.Sequential(
        [
            layers.Dense(hidden_size,
                         input_dim=100,
                         kernel_initializer=init_mode,
                         activation=tf.nn.relu),            
            
            layers.Dropout(0.1),
                        
            layers.Dense(hidden_size,
                         kernel_initializer=init_mode,
                         activation=tf.nn.relu),         
            
            layers.Dense(hidden_size,
                         kernel_initializer=init_mode,
                         activation=tf.nn.relu),             
            
            layers.Dense(num_classes, 
                         kernel_initializer=init_mode, 
                         activation=tf.nn.softmax),
        ]
    )


def create_model_LSTM():
    return keras.Sequential(
        [
            layers.LSTM(2, input_shape=(24, 768)),
            layers.Dropout(0.1),
            layers.Activation('softmax')
        ]
    )


###
# TRAINING EVALUATION
###

def plot_loss(history):
    plt.plot(history.history['loss'], label="train loss")
    plt.plot(history.history['val_loss'], label="val loss")
    plt.legend()
    plt.grid()
    plt.show()


def plot_accuracy(history):
    plt.plot(history.history['accuracy'], label="train accuracy")
    plt.plot(history.history['val_accuracy'], label="val accuracy")
    plt.legend()
    plt.grid()
    plt.show()


def get_confusion_matrix(model, X_test, Y_test):
    Y_pred = model.predict(X_test)
    return confusion_matrix(numpy.argmax(Y_test, axis=1), numpy.argmax(Y_pred, axis=1))
    

def print_metrics(conf_matrix, Y_test):    
    # confusion matrix components
    # [[TN, FP]
    #  [FN, TP]]
    tn = conf_matrix[0,0]
    fn = conf_matrix[1,0]
    fp = conf_matrix[0,1]
    tp = conf_matrix[1,1]
    
    # confusion matrix components as percentages
    tp_percent = tp / len(Y_test) * 100
    tn_percent = tn / len(Y_test) * 100
    fp_percent = fp / len(Y_test) * 100
    fn_percent = fn / len(Y_test) * 100
    
    
    print("Confusion matrix:")
    print(conf_matrix)
    print()
    print("Confusion matrix (Percentages):")
    print(numpy.array([
        [round(tn_percent, 3), round(fp_percent, 3)],
        [round(fn_percent, 3), round(tp_percent, 3)]
    ]))
    
    print("\nMetrics:")
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    
    print("Sensitivity (TPR): {:8.6f}".format( tpr ))
    print("Specifity (TNR):   {:8.6f}".format( tnr ))
    print()
    print("FPR: {:8.6f}".format( fpr ))
    print("FNR: {:8.6f}".format( fnr ))
    print()
    print("Balanced accuracy: {:8.6f}".format( (tpr + tnr) /2 ))
    
    
    
##############################


# https://towardsdatascience.com/keras-data-generators-and-how-to-use-them-b69129ed779c

from tensorflow.keras.utils import Sequence
class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, _X, _y, to_fit=True, 
                 batch_size=256, n_classes=2):
        
        """Initialization
        :param list_IDs: list of all 'label' ids to use in the generator
        :param labels: list of image labels (file names)
        :param to_fit: True to return X and y, False to return X only
        :param batch_size: batch size at each iteration
        :param dim: tuple indicating image dimension
        :param n_classes: number of output masks
        """
        self._X = _X
        self._y = _y
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.on_epoch_end()
    
    
    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(numpy.ceil(self._X.shape[0] / self.batch_size))


    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        #print(f"Returning samples [{index * self.batch_size}:{(index + 1) * self.batch_size}]")
        if self.to_fit:
            return self._X[index * self.batch_size:(index + 1) * self.batch_size, :, :], self._y[index * self.batch_size:(index + 1) * self.batch_size]
        else:
            return self._X[index * self.batch_size:(index + 1) * self.batch_size, :, :]
        
        
        
        
from tensorflow.keras.utils import to_categorical

class ResNetDataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, _X, _y, to_fit=True, 
                 batch_size=256, n_classes=2):
        
        """Initialization
        :param list_IDs: list of all 'label' ids to use in the generator
        :param labels: list of image labels (file names)
        :param to_fit: True to return X and y, False to return X only
        :param batch_size: batch size at each iteration
        :param dim: tuple indicating image dimension
        :param n_classes: number of output masks
        """
        self._X = numpy.expand_dims( _X , axis=3 )
        self._y = _y
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.on_epoch_end()
    
    
    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(numpy.ceil(self._X.shape[0] / self.batch_size))


    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        #print(f"Returning samples [{index * self.batch_size}:{(index + 1) * self.batch_size}]")
        if self.to_fit:
            return self._X[index * self.batch_size:(index + 1) * self.batch_size, :, :], to_categorical( self._y[index * self.batch_size:(index + 1) * self.batch_size] )
        else:
            return self._X[index * self.batch_size:(index + 1) * self.batch_size, :, :]
        
