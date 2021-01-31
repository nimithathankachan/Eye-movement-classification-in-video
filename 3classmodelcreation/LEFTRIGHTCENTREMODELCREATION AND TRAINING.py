import os
import numpy as np
import shutil
import cv2
from google.colab.patches import cv2_imshow

#DATASET PARTITIONING

root_dir='resized eyes'
centreposCls = '/resizedcentreimages' 
leftposCls = '/resizedleftimages' 
rightposCls = '/resizedrightimages' 

os.makedirs('Train'+ centreposCls)
os.makedirs('Train'+ leftposCls)
os.makedirs('Train'+ rightposCls)
os.makedirs('Validation'+ centreposCls)
os.makedirs('Validation'+ leftposCls)
os.makedirs('Validation'+ rightposCls)
os.makedirs('Test'+ centreposCls)
os.makedirs('Test'+ leftposCls)
os.makedirs('Test'+ rightposCls)

os.makedirs('Train1'+ centreposCls)
os.makedirs('Train1'+ leftposCls)
os.makedirs('Train1'+ rightposCls)
os.makedirs('Validation1'+ centreposCls)
os.makedirs('Validation1'+ leftposCls)
os.makedirs('Validation1'+ rightposCls)
os.makedirs('Test1'+ centreposCls)
os.makedirs('Test1'+ leftposCls)
os.makedirs('Test1'+ rightposCls)


currentCls=centreposCls
src='/content/drive/My Drive/FINAL2'+currentCls 
allImageNames=os.listdir(src) 
np.random.shuffle(allImageNames)
training_imagenames,validation_imagenames,test_imagenames=np.split(np.array(allImageNames),
				[int(len(allImageNames)*0.8),int(len(allImageNames)*0.9)])
training_imagenames=[src+'/'+ name for name in training_imagenames.tolist()] 
validation_imagenames=[src+'/'+ name for name in validation_imagenames.tolist()]
test_imagenames=[src+'/'+ name for name in test_imagenames.tolist()]
print('Total images:',len(allImageNames))
print('Training:',len(training_imagenames))
print('Validation:',len(validation_imagenames))
print('Testing:',len(test_imagenames))

for name in training_imagenames:
    shutil.copy(name,'Train/'+ currentCls)
for name in validation_imagenames:
    shutil.copy(name,'Validation/'+ currentCls)
for name in test_imagenames:
    shutil.copy(name,'Test/'+ currentCls)

currentCls=leftposCls
src='/content/drive/My Drive/FINAL2'+currentCls 
allImageNames=os.listdir(src) 
np.random.shuffle(allImageNames) 
training_imagenames,validation_imagenames,test_imagenames=np.split(np.array(allImageNames),
				[int(len(allImageNames)*0.8),int(len(allImageNames)*0.9)])
training_imagenames=[src+'/'+ name for name in training_imagenames.tolist()] 
validation_imagenames=[src+'/'+ name for name in validation_imagenames.tolist()]
test_imagenames=[src+'/'+ name for name in test_imagenames.tolist()]

print('Total images:',len(allImageNames))
print('Training:',len(training_imagenames))
print('Validation:',len(validation_imagenames))
print('Testing:',len(test_imagenames))

for name in training_imagenames:
    shutil.copy(name,'Train/'+ currentCls)
for name in validation_imagenames:
    shutil.copy(name,'Validation/'+ currentCls)
for name in test_imagenames:
    shutil.copy(name,'Test/'+ currentCls)


currentCls=rightposCls
src='/content/drive/My Drive/FINAL2/resizedrightimages'
allImageNames=os.listdir(src) 
np.random.shuffle(allImageNames) 
training_imagenames,validation_imagenames,test_imagenames=np.split(np.array(allImageNames),
				[int(len(allImageNames)*0.8),int(len(allImageNames)*0.9)])
training_imagenames=[src+'/'+ name for name in training_imagenames.tolist()]
validation_imagenames=[src+'/'+ name for name in validation_imagenames.tolist()]
test_imagenames=[src+'/'+ name for name in test_imagenames.tolist()]

print('Total images:',len(allImageNames))
print('Training:',len(training_imagenames))
print('Validation:',len(validation_imagenames))
print('Testing:',len(test_imagenames))

for name in training_imagenames:
    shutil.copy(name,'Train/'+ currentCls)
for name in validation_imagenames:
    shutil.copy(name,'Validation/'+ currentCls)
for name in test_imagenames:
    shutil.copy(name,'Test/'+ currentCls)



directory = '/content/Train/resizedcentreimages'
for file in os.listdir(directory):
  img_path = os.path.join(directory, file)
  img = cv2.imread(img_path,0)
  img = cv2.resize(img, (50, 50))  
  equ = cv2.equalizeHist(img)
  cv2.imwrite('/content/Train1/resizedcentreimages/' +  file + '_hist.jpg', equ) 


directory = '/content/Train/resizedleftimages'
for file in os.listdir(directory):
  img_path = os.path.join(directory, file)
  img = cv2.imread(img_path,0)
  img = cv2.resize(img, (50, 50))
  equ = cv2.equalizeHist(img)
  cv2.imwrite('/content/Train1/resizedleftimages/' +  file + '_hist.jpg', equ) 


directory = '/content/Train/resizedrightimages'
for file in os.listdir(directory):
  img_path = os.path.join(directory, file)
  img = cv2.imread(img_path,0)
  img = cv2.resize(img, (50, 50))
  equ = cv2.equalizeHist(img)
  cv2.imwrite('/content/Train1/resizedrightimages/' +  file + '_hist.jpg', equ)

 
directory = '/content/Validation/resizedcentreimages'
for file in os.listdir(directory):
  img_path = os.path.join(directory, file)
  img = cv2.imread(img_path,0) 
  img = cv2.resize(img, (50, 50))
  equ = cv2.equalizeHist(img)
  cv2.imwrite('/content/Validation1/resizedcentreimages/' +  file + '_hist.jpg', equ) 


directory = '/content/Validation/resizedrightimages'
for file in os.listdir(directory):
  img_path = os.path.join(directory, file)
  img = cv2.imread(img_path,0) 
  img = cv2.resize(img, (50, 50))
  equ = cv2.equalizeHist(img)
  cv2.imwrite('/content/Validation1/resizedrightimages/' +  file + '_hist.jpg', equ) 


directory = '/content/Validation/resizedleftimages'
for file in os.listdir(directory):
  img_path = os.path.join(directory, file)
  img = cv2.imread(img_path,0)
  img = cv2.resize(img, (50, 50))
  equ = cv2.equalizeHist(img)
  cv2.imwrite('/content/Validation1/resizedleftimages/' +  file + '_hist.jpg', equ) 


directory = '/content/Test/resizedrightimages'
for file in os.listdir(directory):
  img_path = os.path.join(directory, file)
  img = cv2.imread(img_path,0) 
  img = cv2.resize(img, (50, 50))
  equ = cv2.equalizeHist(img)
  cv2.imwrite('/content/Test1/resizedrightimages/' +  file + '_hist.jpg', equ) 


directory = '/content/Test/resizedleftimages'
for file in os.listdir(directory):
  img_path = os.path.join(directory, file)
  img = cv2.imread(img_path,0)
  img = cv2.resize(img, (50, 50))
  equ = cv2.equalizeHist(img)
  cv2.imwrite('/content/Test1/resizedleftimages/' +  file + '_hist.jpg', equ)


directory = '/content/Test/resizedcentreimages'
for file in os.listdir(directory):
  img_path = os.path.join(directory, file)
  img = cv2.imread(img_path,0)
  img = cv2.resize(img, (50, 50))
  equ = cv2.equalizeHist(img)
  cv2.imwrite('/content/Test1/resizedcentreimages/' +  file + '_hist.jpg', equ)


#MODEL CREATION AND TRAINING

import keras
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Input
from keras.layers import Dense, Dropout, Activation, Flatten
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot


lenet = Sequential()
lenet.add(Conv2D(24, kernel_size=(7, 7), activation='relu',
                 input_shape=(50,50,3)))
lenet.add(MaxPooling2D(pool_size=(2, 2)))
lenet.add(Conv2D(24, (5, 5), activation='relu'))
lenet.add(MaxPooling2D(pool_size=(2, 2)))
lenet.add(Conv2D(24, (3, 3), activation='relu'))
lenet.add(MaxPooling2D(pool_size=(2, 2)))
lenet.add(Flatten())
lenet.add(Dense(500, activation='relu'))
lenet.add(Dense(3, activation='softmax'))

SVG(model_to_dot(lenet,show_shapes=True, show_layer_names=False
                ).create(prog='dot', format='svg'))
lenet.summary()
lenet.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])



from keras.preprocessing.image import ImageDataGenerator 
train_datagen=ImageDataGenerator(rescale=1/255,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 brightness_range=[0.2,1.0],
                                 vertical_flip=True)


validation_datagen=ImageDataGenerator(rescale=1/255)
test_datagen = ImageDataGenerator(rescale=1/255)
batchSize=5
training_set=train_datagen.flow_from_directory(  
                           '/content/Train1',
                            target_size=(50,50),
                            batch_size=batchSize,
                            class_mode='categorical')
validation_set=validation_datagen.flow_from_directory( 
                           '/content/Validation1',
                            target_size=(50,50),
                            batch_size=batchSize,
                            class_mode='categorical')
test_set = test_datagen.flow_from_directory(
                           '/content/Test1',
                            target_size=(50,50),
                            batch_size=batchSize,
                            class_mode='categorical')
stepsnumperepochtraining=int(408/batchSize)
stepsnumperepochvalidation=int(59/batchSize)
stepsnumperepochtesting = int(118/batchSize)
history = lenet.fit_generator(         
            training_set,
            steps_per_epoch = stepsnumperepochtraining,
            epochs = 50,
            validation_data = validation_set,
            validation_steps = stepsnumperepochvalidation)

lenet.save('eyeprojectmodel2.h5')
from shutil import copyfile
copyfile('/content/eyeprojectmodel2.h5','/content/drive/My Drive/eyeprojectmodel/projectmodel2.h5') 





 

