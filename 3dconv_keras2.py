import numpy as np
from astropy.io import fits
import glob
import numpy
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam

from keras.layers import (Activation, Conv3D, Dense, Dropout, Flatten,
                          MaxPooling3D)

from keras.optimizers import SGD, RMSprop
from keras.utils import np_utils, generic_utils

import theano
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import gc

from sklearn.cross_validation import train_test_split
from sklearn import cross_validation
from sklearn import preprocessing

img_rows=10
img_cols = 10
img_depth = 500

patch_size = 500

single = glob.glob('/lustre/pipeline/scratch/KEYSTONE/3d_gauss_train/*.fits')
multi = glob.glob('/lustre/pipeline/scratch/KEYSTONE/3d_gauss_train_multi/*.fits')

def get_train_set(single, multi):
	train_data = []
	labels = []
	count=0
	for i in single:
		ss = fits.open(i, memmap=False)
		train_data.append(ss[0].data)
		labels.append(0)
		ss.close()
		del ss
		gc.collect()
		count+=1
		print count

	for i in multi:
		ss = fits.open(i, memmap=False)
		train_data.append(ss[0].data)
		labels.append(1)
		ss.close()
		del ss
		gc.collect()
	return numpy.array(train_data), numpy.array(labels)

X_train_new, y_train_new = get_train_set(single[0:5000], multi[0:5000])
X_val_new, y_val_new = get_train_set(single[-100:], multi[-100:])

# CNN Training parameters

batch_size = 100
nb_classes = 2
nb_epoch = 50

print X_train_new.reshape(X_train_new.shape[0], img_rows, img_cols, img_depth, 1)
input_shape = (img_rows, img_cols, img_depth, 1)
# convert class vectors to binary class matrices
X_train_new = X_train_new.reshape(X_train_new.shape[0], img_rows, img_cols, img_depth, 1)
y_train_new = np_utils.to_categorical(y_train_new, nb_classes)

X_val_new = X_val_new.reshape(X_val_new.shape[0], img_rows, img_cols, img_depth, 1)
y_val_new = np_utils.to_categorical(y_val_new, nb_classes)

# Define model
print (img_rows, img_cols, img_depth, 1)
# Define model
model = Sequential()
print X_train_new[0].shape
model.add(Conv3D(32, kernel_size=(2, 2, 2), input_shape=(img_rows, img_cols, img_depth, 1), padding='same'))
model.add(Activation('relu'))
model.add(Conv3D(32, kernel_size=(2, 2, 2), padding='same'))
model.add(Activation('softmax'))
model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='same'))
model.add(Dropout(0.25))

model.add(Conv3D(64, kernel_size=(2, 2, 2), padding='same'))
model.add(Activation('relu'))
model.add(Conv3D(64, kernel_size=(2, 2, 2), padding='same'))
model.add(Activation('softmax'))
model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='same'))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes, activation='softmax'))

model.compile(loss=categorical_crossentropy,
                  optimizer=Adam(), metrics=['accuracy'])
model.summary()

  
# Split the data

#X_train_new, X_val_new, y_train_new,y_val_new =  train_test_split(train_set, Y_train, test_size=0.2, random_state=4)


# Train the model

hist = model.fit(X_train_new, y_train_new, validation_data=(X_val_new,y_val_new),
          batch_size=batch_size,epochs = nb_epoch,shuffle=True)


#hist = model.fit(train_set, Y_train, batch_size=batch_size,
#         nb_epoch=nb_epoch,validation_split=0.2, show_accuracy=True,
#           shuffle=True)


 # Evaluate the model
score = model.evaluate(X_val_new, y_val_new, batch_size=batch_size)
print('Test score:', score[0])
print('Test accuracy:', score[1]) 


