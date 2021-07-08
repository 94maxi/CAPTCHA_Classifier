
# ===============================================================================================================
# # Task 1. 
# ===============================================================================================================

#package
import tensorflow
import numpy as np
import sklearn
import keras as keras
import tensorflow as tf
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten,Activation, Conv2D, MaxPooling2D, BatchNormalization
from keras.callbacks import LearningRateScheduler,History, EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from keras.layers import GaussianNoise #task 2

# ===============================================================================================================
#load data
with np.load("training-dataset.npz") as data:
    img = data["x"]
    lbl = data["y"]
# Make it 0 based indices
lbl = lbl-1


# train- val - test - random
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(img, lbl, test_size=0.10, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=(0.10 / 0.90), random_state=1)

# ===============================================================================================================
# Feature engineering
# Convert class vectors to binary class matrices
# change the type as float
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_val = X_val.astype("float32")

# Normalize the data 
X_train /= 255
X_test /= 255
X_val /= 255

# One-hot coding
num_classes = 26
Y_train = keras.utils.to_categorical(y_train, num_classes)
Y_test = keras.utils.to_categorical(y_test, num_classes)
Y_val = keras.utils.to_categorical(y_val, num_classes)

# ===============================================================================================================
#Baseline scores
# from sklearn.metrics import accuracy_score
# from sklearn.linear_model import LogisticRegression
# baseline = LogisticRegression()
# baseline.fit(X_train, y_train)
# print("Error rate of logistic regression:",1-accuracy_score(y_val, baseline.predict(X_val)))
# #Error rate of logistic regression: 0.28309294871794877

# ===============================================================================================================
# # Model 1 Neural network 1.0
# ===============================================================================================================
# --------------------------------------------------------------------------------------------------------------
# # Model 1 - Experiment 1 - Not adapated - build the basic modle and found that epochs size is too much 100
# --------------------------------------------------------------------------------------------------------------

# model = Sequential()
# model.add(Dense(128, input_dim=X_train.shape[1], activation='relu')) 
# model.add(Dense(128, activation='relu')) 
# model.add(Dense(Y_train.shape[1], activation='softmax')) 

# optimizer = Adam(lr=0.001)
# # For classification, the loss function should be categorical_crossentropy
# model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
# # model.fit(X_train, Y_train, epochs=100, batch_size=16, verbose=0)

# history = model.fit(X_train, Y_train,
#                     validation_data=(X_val, Y_val), epochs=100, batch_size=32, verbose=1)
# --------------------------------------------------------------------------------------------------------------
# Model 1 Experiment 3 - Not adapated -  Neural network Test learning rate differences
# --------------------------------------------------------------------------------------------------------------
# i = [0.0001,0.0005,0.0006,0.0007,0.0008,0.0009]
# for k in i:
#     optimizer = keras.optimizers.Adam(lr=k)
#     model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

#     history = model.fit(X_train, Y_train,validation_data=(X_val, Y_val) ,epochs=8, batch_size=32, verbose=1)

# --------------------------------------------------------------------------------------------------------------
# Model 1 - Experiment 2 - Adpated - more layer with setting avoid overfitting 
# --------------------------------------------------------------------------------------------------------------
model = Sequential()

model.add(Dense(256, input_dim=X_train.shape[1], activation='relu')) #change from tahn to relu
model.add(Dense(256, activation='relu')) #change from tahn to relu
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(128, input_dim=X_train.shape[1], activation='relu')) #change from tahn to relu
model.add(Dense(128, activation='relu')) #change from tahn to relu
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(Y_train.shape[1], activation='softmax')) # We need to have as many units as classes,

optimizer = keras.optimizers.Adam(lr=0.001)
#For classification, the loss function should be categorical_crossentropy

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# #earlystopping epochs=9:
# from keras import callbacks
# earlystopping = callbacks.EarlyStopping(monitor ="val_loss",
#                                         mode ="min", patience = 3,
#                                         restore_best_weights = True,
#                                         verbose = 1)
# history = model.fit(X_train, Y_train,validation_data=(X_val, Y_val) ,epochs=100, batch_size=32, callbacks=[earlystopping])
#

history = model.fit(X_train, Y_train,validation_data=(X_val, Y_val) ,epochs=9, batch_size=32, verbose=1)


# --------------------------------------------------------------------------------------------------------------
# Model 1 - Test accuracy
scores = model.evaluate(X_test, Y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100)) # Test Accuracy: 91.15%


# ===============================================================================================================
# # Model 2 Neural network 1.0
# ===============================================================================================================
# --------------------------------------------------------------------------------------------------------------
# Prepocessing
# --------------------------------------------------------------------------------------------------------------
#data preprocess stages - reshaping to 4 dimensionals
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_val = X_val.reshape(X_val.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

# --------------------------------------------------------------------------------------------------------------
# Model 1 - Basic CNN model - not adapted
# --------------------------------------------------------------------------------------------------------------
# # Creating a Sequential Model and adding the layers
# model = Sequential()
# model.add(Conv2D(28, kernel_size=(8,4), input_shape=input_shape)) # 28,8,4
# model.add(MaxPooling2D(pool_size=(4,6))) #4,6
# model.add(Flatten()) # flatten 2D arrays for fully connected layers
# model.add(Dense(128, activation=tf.nn.relu)) #relu activation function
# model.add(Dropout(0.2)) 
# model.add(Dense(26,activation=tf.nn.softmax)) #multiclass function softmax

# # Compile the model
# model.compile(optimizer='adam', #from adam
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])

# history = model.fit(X_train, Y_train,validation_data=(X_val, Y_val) ,epochs=10, batch_size=20, verbose=1)

#----------------------------------------------------------------------------------------------------------------
# Model 2 - Experiment 2 - Early stopping strategy - Adapted
# ----------------------------------------------------------------------------------------------------------------
# # BUILD CONVOLUTIONAL NEURAL NETWORKS
model = Sequential()

model.add(Conv2D(32, kernel_size = 3, activation='relu', input_shape = (28, 28, 1)))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size = 3, activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size = 5, strides=2, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4)) 

model.add(Conv2D(64, kernel_size = 3, activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size = 3, activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size = 5, strides=2, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4)) 

model.add(Conv2D(128, kernel_size = 4, activation='relu'))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dropout(0.4)) 
model.add(Dense(26, activation='softmax'))

#COMPILE WITH ADAM OPTIMIZER AND CROSS ENTROPY COST
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# #earlystopping:
# from keras import callbacks
# earlystopping = callbacks.EarlyStopping(monitor ="val_loss",
#                                         mode ="min", patience = 3,
#                                         restore_best_weights = True,
#                                         verbose = 1)
# history = model.fit(X_train, Y_train,validation_data=(X_val, Y_val) ,epochs=100, batch_size=64, callbacks=[earlystopping])

history = model.fit(X_train, Y_train,validation_data=(X_val, Y_val) ,epochs=15, batch_size=64, verbose=1)

#----------------------------------------------------------------------------------------------------------------
# Model 2 - Experiment 3 - LearningRateScheduler - Not adapted
# ----------------------------------------------------------------------------------------------------------------
# BUILD CONVOLUTIONAL NEURAL NETWORKS

# model = Sequential()

# model.add(Conv2D(32, kernel_size = 3, activation='relu', input_shape = (28, 28, 1)))
# model.add(BatchNormalization())
# model.add(Conv2D(32, kernel_size = 3, activation='relu'))
# model.add(BatchNormalization())
# model.add(Conv2D(32, kernel_size = 5, strides=2, padding='same', activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.4)) 

# model.add(Conv2D(64, kernel_size = 3, activation='relu'))
# model.add(BatchNormalization())
# model.add(Conv2D(64, kernel_size = 3, activation='relu'))
# model.add(BatchNormalization())
# model.add(Conv2D(64, kernel_size = 5, strides=2, padding='same', activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.4)) 

# model.add(Conv2D(128, kernel_size = 4, activation='relu'))
# model.add(BatchNormalization())
# model.add(Flatten())
# model.add(Dropout(0.4)) 
# model.add(Dense(26, activation='softmax'))

# #COMPILE WITH ADAM OPTIMIZER AND CROSS ENTROPY COST
# model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# #DECREASE LEARNING RATE BY 0.95 EACH EPOCH
# annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)

# # # TRAIN CNNs AND DISPLAY ACCURACIES
# epochs = 30

# from keras import callbacks 
# history = model.fit(X_train,Y_train, batch_size=64, epochs = epochs, 
#                    validation_data = (X_val,Y_val), callbacks=[annealer], verbose=1)


# =============================================================================
#Plot Accuracy
# import matplotlib.pyplot as plt
# 
# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# 
# epochs = range(1, len(acc) + 1)
# plt.figure(figsize=(5,5))
# plt.plot(epochs, loss, label='Training loss')
# plt.plot(epochs, val_loss, label='Validation loss')
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# 
# plt.show()
# 
# plt.clf()
# plt.figure(figsize=(5,5))
# plt.plot(epochs, acc, label='Training acc')
# plt.plot(epochs, val_acc, label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# 
# 
# plt.show()
# =============================================================================

# --------------------------------------------------------------------------------------------------------------
# Model 2 - Test accuracy
scores = model.evaluate(X_test, Y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100)) #Accuracy: 95.41%



