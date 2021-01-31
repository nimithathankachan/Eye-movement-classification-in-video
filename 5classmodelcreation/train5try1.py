import csv
import numpy as np 
import keras 
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D,Flatten,Dense,Activation,Dropout,MaxPooling2D
from keras.activations import relu
from keras.optimizers import Adam






model = Sequential()

model.add(Conv2D(32, (3,3), padding = 'same', input_shape=(50,50,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64, (2,2), padding= 'same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (2,2), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(5))
model.add(Activation('softmax'))

	
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy',metrics=['accuracy'])




train_datagen = ImageDataGenerator(rescale=1/255,
        rotation_range=2,
        width_shift_range=0.2,
        height_shift_range=0.2,
	shear_range=0.2,
        zoom_range=0.2,
        brightness_range=[0.2,1.0]
        )


validation_datagen=ImageDataGenerator(rescale=1/255)
test_datagen = ImageDataGenerator(rescale=1/255)
batchSize=5
training_set=train_datagen.flow_from_directory(   ##now training_set will contain images from Train folder. its size will be 50*50.
                           'Train1',
                            target_size=(50,50),
                            batch_size=batchSize,
                            class_mode='categorical')
validation_set=validation_datagen.flow_from_directory(  #now  validation_set will contain images from Validationfolder and size will be 50*50.
                           'Validation1',
                            target_size=(50,50),
                            batch_size=batchSize,
                            class_mode='categorical')
test_set = test_datagen.flow_from_directory(
                           'Test1',
                            target_size=(50,50),
                            batch_size=batchSize,
                            class_mode='categorical')


stepsnumperepochtraining=int(2140/batchSize)
stepsnumperepochvalidation=int(270/batchSize)
stepsnumperepochtesting = int(270/batchSize)



history = model.fit_generator(          #history is object that is used to plot graph between accuracy and loss.
            training_set,
            steps_per_epoch = stepsnumperepochtraining,
            epochs = 200,
            validation_data = validation_set,
            validation_steps = stepsnumperepochvalidation)


#model.save('5class5.hdf5')





