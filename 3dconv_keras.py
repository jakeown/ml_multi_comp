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

from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

from keras.layers import (Activation, Conv3D, Dense, Dropout, Flatten,
                          MaxPooling3D)

from keras.optimizers import SGD, RMSprop
from keras.utils import np_utils, generic_utils

import theano
import os

import numpy as np
import gc
import sys
import h5py

from sklearn.cross_validation import train_test_split
from sklearn import cross_validation
from sklearn import preprocessing

import struct
import numpy

img_rows=10
img_cols = 10
img_depth = 500

patch_size = 500

#single = glob.glob('3d_gauss_train/*.fits')
#multi = glob.glob('3d_gauss_train_multi/*.fits')

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
		print str(count) + ' samples completed \r',
		sys.stdout.flush()

	for i in multi:
		ss = fits.open(i, memmap=False)
		train_data.append(ss[0].data)
		labels.append(1)
		ss.close()
		del ss
		gc.collect()
		print str(count) + ' samples completed \r',
		sys.stdout.flush()
	return numpy.array(train_data), numpy.array(labels)

def get_train_set2(start=0, length=200):
	print 'Loading Training Data...'
	with h5py.File('training.h5', 'r') as hf:
		X = hf['data'][:]
	hf.close()
	with h5py.File('labels.h5', 'r') as hf:
		y = hf['data'][:]
	hf.close()
	return X, y

def get_val_set2(start=0, length=200):
	print 'Loading Training Data...'
	with h5py.File('testing.h5', 'r') as hf:
		X = hf['data'][:]
	hf.close()
	with h5py.File('test_labels.h5', 'r') as hf:
		y = hf['data'][:]
	hf.close()
	return X, y

X_train_new, y_train_new = get_train_set2()
print X_train_new.shape

# CNN Training parameters

#batch_size = 2
#nb_classes = 2
#nb_epoch =50

# convert class vectors to binary class matrices
X_train_new = X_train_new.reshape(X_train_new.shape[0], img_rows, img_cols*img_depth)

# Define model
model = Sequential()
#model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(LSTM(150, input_shape = (10, 5000), return_sequences=True))
model.add(LSTM(150, return_sequences=True))
model.add(LSTM(150, return_sequences=False))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print numpy.shape(X_train_new)
print numpy.shape(y_train_new)
model.fit(X_train_new, y_train_new, epochs=10, batch_size=64)
# Final evaluation of the model
X_val_new, y_val_new = get_val_set2()
X_val_new = X_val_new.reshape(X_val_new.shape[0], img_rows, img_cols*img_depth)
scores = model.evaluate(X_val_new, y_val_new, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

model.save("model_3layer_40000.h5")
print("Saved model to disk")
