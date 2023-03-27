#  # Importing the required libraries
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# # Setting the image size and batch size
# img_height = 224
# img_width = 224
# batch_size = 32

# # Setting up the training and validation data directories
# # D:\project\kaggle\colored_images\Mild
# train_dir = "D:\\project\\kaggle\\colored_images\\Mild"
# validation_dir = "D:\\project\\kaggle\\colored_images\\Mild"

# # Setting up the data augmentation
# train_datagen = ImageDataGenerator(rescale=1./255,
#                                    shear_range=0.2,
#                                    zoom_range=0.2,
#                                    horizontal_flip=True)
# validation_datagen = ImageDataGenerator(rescale=1./255)

# # Setting up the data generators
# train_generator = train_datagen.flow_from_directory(train_dir,
#                                                     target_size=(img_height, img_width),
#                                                     batch_size=batch_size,
#                                                     class_mode='binary')
# validation_generator = validation_datagen.flow_from_directory(validation_dir,
#                                                               target_size=(img_height, img_width),
#                                                               batch_size=batch_size,
#                                                               class_mode='binary')

# # Setting up the CNN model
# model = keras.Sequential([
#     layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(64, (3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(128, (3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#     layers.Flatten(),
#     layers.Dense(128, activation='relu'),
#     layers.Dense(1, activation='sigmoid')
# ])

# # Compiling the model
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# # Training the model
# model.fit(train_generator, validation_data=validation_generator, epochs=10)

# =====================================================================================================================================================

# Import libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# Set up data generators
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'D:/project/kaggle/colored_images',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training')

validation_generator = train_datagen.flow_from_directory(
    'D:/project/kaggle/colored_images',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation')

test_generator = test_datagen.flow_from_directory(
    'D:\\project\\kaggle\\colored_images',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary')

model = Sequential()
model.add(Conv2D(64,(3,3),input_shape = (224,224,3)))      # regularizing datasets
model.add(Activation("relu"))                 # relu removes the negative part
model.add(MaxPooling2D(pool_size =(2,2)))         # It reduces the amount of parameter and computation in the network
model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size =(2,2)))
model.add(Dropout(0.3))           # It reduces overfitting and increases the accuracy of the training datasets
model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size =(2,2)))
model.add(Dropout(0.5))
model.add(Flatten())     #  It convert the datasets into 1D array
model.add(Dense(64))          
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation("sigmoid")) 

# Build CNN model
# model = Sequential()
# model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
# model.add(MaxPooling2D((2, 2)))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D((2, 2)))
# model.add(Conv2D(128, (3, 3), activation='relu'))
# model.add(MaxPooling2D((2, 2)))
# model.add(Conv2D(256, (3, 3), activation='relu'))
# model.add(MaxPooling2D((2, 2)))
# model.add(Flatten())
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(train_generator, epochs=10, validation_data=validation_generator)

# Evaluate model on test set
test_loss, test_acc = model.evaluate(test_generator)
print('Test accuracy:', test_acc)

# Make predictions on test set
predictions = model.predict(test_generator)
predicted_classes = predictions > 0.5

# Print classification report and confusion matrix
print(classification_report(test_generator.classes, predicted_classes))
print(confusion_matrix(test_generator.classes, predicted_classes))
model.save('retinopathy.model')
