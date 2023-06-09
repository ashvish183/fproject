#importing tensorflow libraries
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D , MaxPooling2D , Dense , Activation , Dropout , Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import pickle 
import numpy as np
import os
# loading training dataset using pickle 
pickle_in_x = open("xe.pickle","rb")
xe = pickle.load(pickle_in_x)
pickle_in_y = open("ye.pickle","rb")
ye = pickle.load(pickle_in_y)
xe = xe/255.0
# training the dataset using convolutional Neural Network
# model = Sequential()
# model.add(Conv2D(64,(3,3),input_shape = xe.shape[1:]))      # regularizing datasets
# model.add(Activation("relu"))                 # relu removes the negative part
# model.add(MaxPooling2D(pool_size =(2,2)))         # It reduces the amount of parameter and computation in the network
# model.add(Conv2D(64,(3,3)))
# model.add(Activation("relu"))
# model.add(MaxPooling2D(pool_size =(2,2)))
# model.add(Dropout(0.3))           # It reduces overfitting and increases the accuracy of the training datasets
# model.add(Conv2D(64,(3,3)))
# model.add(Activation("relu"))
# model.add(MaxPooling2D(pool_size =(2,2)))
# model.add(Dropout(0.5))
# model.add(Flatten())     #  It convert the datasets into 1D array
# model.add(Dense(64))          
# model.add(Activation("relu"))
# model.add(Dropout(0.5))
# model.add(Dense(1))
# model.add(Activation("sigmoid")) 

# model.compile(loss='binary_crossentropy',optimizer = "adam",metrics = ['accuracy'])
# print(ye)
# model.fit(xe,ye,batch_size=9,epochs = 14)   
# model.save('retinopathy.model') # saving the model




model = Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=xe.shape[1:]),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compiling the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the model
model.fit(xe,ye,epochs=10)
model.save('retinopathy.model') # saving the model

