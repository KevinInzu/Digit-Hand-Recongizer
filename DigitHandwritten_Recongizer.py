# Handwritten digit recognition for MNIST dataset using Convolution Neural Networks

import tensorflow as tf
from keras.datasets import mnist # This is used to load mnist dataset later
from keras.utils import np_utils # This will be used to convert your test image to a categorical class (digit from 0 to 9)
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten

# Load and return training and test datasets
def load_dataset():
	# Load dataset X_train, X_test, y_train, y_test via imported keras library
	(X_train,y_train), (X_test,y_test) = mnist.load_data();
	# reshape for X train and test vars - Hint: X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32')
	X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32')
	X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32')
 			
	# normalize inputs from 0-255 to 0-1 - Hint: X_train = X_train / 255
	
	X_train /= 255
	X_test /= 255
	
	# Convert y_train and y_test to categorical classes
	y_train = np_utils.to_categorical(y_train)
	y_test = np_utils.to_categorical(y_test)
	
	# return your X_train, X_test, y_train, y_test
	return X_train,X_test,y_train,y_test

def digit_recognition_cnn():
	cnn = Sequential()
	# Conv + ReLU + Flatten + Dense layers
	cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=5, activation='relu')) 
	cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
	cnn.add(tf.keras.layers.Conv2D(filters=15, kernel_size=3, activation='relu'))
	cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
	cnn.add(Dropout(0.2))
	cnn.add(Flatten())
	cnn.add(Dense(units = 128, activation = 'relu'))
	cnn.add(Dense(units = 50, activation = 'relu'))
	cnn.add(Dense(units = 10, activation = 'softmax'))
	# 3b. Compile your model with categorical_crossentropy (loss), adam optimizer and accuracy as a metric
	cnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# 3c. return your model
	return cnn

model = digit_recognition_cnn()
#epochs to a number between 10 - 20 and batch_size between 150 - 200
X_train,X_test,y_train,y_test = load_dataset()
model.fit(X_train, y_train, validation_data = (X_test,y_test),batch_size = 200, epochs = 20)
print( model.evaluate(X_test,y_test) )

model.save('digitRecognizer.h5')


# Code below to make a prediction for a new image.

from keras.utils import load_img
from keras.utils  import img_to_array
from keras.models import load_model
import numpy as np
 
def load_new_image(path):
	newImage = load_img(path, grayscale=True, target_size=(28, 28))
	newImage = img_to_array(newImage)
	newImage = newImage.reshape((1, 28, 28, 1)).astype('float32')
	newImage = newImage / 255
	return newImage
 
def test_model_performance():
	img = load_new_image('number_9.jpg')
	model  = load_model('digitRecognizer.h5')
	imageClass =  np.argmax(model.predict(img), axis=-1)
	#Print prediction result
	print(imageClass[0])
 
test_model_performance()