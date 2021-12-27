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
    
    return numpy.vstack(dataset_balanced.X), dataset_balanced.y
    
    
def oversample_smote(X, y):
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


def create_model_MLP_CWE(num_classes, init_mode='glorot_uniform'):
    return keras.Sequential(
        [
            layers.Dense(100,
                         input_dim=100
                         )
        ]
    )

def create_model_MLP_big(init_mode='glorot_uniform', num_classes=2):
    return keras.Sequential(
        [
            layers.Dense(500,
                         input_dim=768,
                         kernel_initializer=init_mode,
                         activation=tf.nn.relu),
            
            
            layers.Dropout(0.1),
                        
            layers.Dense(100,
                         kernel_initializer=init_mode,
                         activation=tf.nn.relu),
            
            layers.Dense(100,
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
    print("False positive rate overall: {:8.6f}".format( conf_matrix[0,1] / len(Y_test) ))
    print("False negative rate overall: {:8.6f}".format( conf_matrix[1,0] / len(Y_test) ))
    print()
    print("False positive / Total positive: {:8.6f}".format( conf_matrix[0,1] / (conf_matrix[0,0] + conf_matrix[0,1]) ))
    print("False negative / Total negative: {:8.6f}".format( conf_matrix[1,0] / (conf_matrix[1,0] + conf_matrix[1,1]) ))
