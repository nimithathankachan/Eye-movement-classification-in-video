#*******Program for splitting the dataset********

import os
import numpy as np
import shutil
import cv2

openposCls = '/opennew' #centreposCls is centre images
closedposCls = '/closednew' #leftposCls is left images

#now create train folder and inside it create open and close folders.Do the same with test and validation folders too.
os.makedirs('Train'+ openposCls)
os.makedirs('Train'+ closedposCls)

os.makedirs('Validation'+ openposCls)
os.makedirs('Validation'+ closedposCls)

os.makedirs('Test'+ openposCls)
os.makedirs('Test'+ closedposCls)




os.makedirs('Train1'+ openposCls)
os.makedirs('Train1'+ closedposCls)
#os.makedirs('Train1'+ rightposCls)
os.makedirs('Validation1'+ openposCls)
os.makedirs('Validation1'+ closedposCls)
#os.makedirs('Validation1'+ rightposCls)
os.makedirs('Test1'+ openposCls)
os.makedirs('Test1'+ closedposCls)
#os.makedirs('Test1'+ rightposCls)


#here all images are considered as the files inside the directories 
currentCls=openposCls
src='opennew' 
allImageNames=os.listdir(src) # here all the image names will be inside a list named allImageNames.
np.random.shuffle(allImageNames) #shuffle the list containing all the images.
training_imagenames,validation_imagenames,test_imagenames=np.split(np.array(allImageNames),[int(len(allImageNames)*0.8),int(len(allImageNames)*0.9)])
#in the above code we splitted the whole  set ie allImageNames into 3 for training ,testing,validation.  np.split[0.7,0.8] means 1st part upto 80%, 2nd part from 80% to 90% , and 3rd part from 90% to full.

training_imagenames=[src+'/'+ name for name in training_imagenames.tolist()] # here we are creating 3 sublists training_imagenames,validation_imagenames,test_imagenames.

validation_imagenames=[src+'/'+ name for name in validation_imagenames.tolist()]

test_imagenames=[src+'/'+ name for name in test_imagenames.tolist()]

print('Total images:',len(allImageNames))
print('Training:',len(training_imagenames))
print('Validation:',len(validation_imagenames))
print('Testing:',len(test_imagenames))

#Now copying into train,test and validation folders

for name in training_imagenames:
    shutil.copy(name,'Train/'+ currentCls)
for name in validation_imagenames:
    shutil.copy(name,'Validation/'+ currentCls)
for name in test_imagenames:
    shutil.copy(name,'Test/'+ currentCls)




currentCls=closedposCls
src='closednew' #here source becomes inside cell_images,the currentCls ie negCls.
allImageNames=os.listdir(src) # here all the image names will be inside a list named allImageNames.
np.random.shuffle(allImageNames) #shuffle the list containing all the images.
training_imagenames,validation_imagenames,test_imagenames=np.split(np.array(allImageNames),[int(len(allImageNames)*0.8),int(len(allImageNames)*0.9)])
#in the above code we splitted the whole uninfecteded set ie allImageNames into 3 for training ,testing,validation.  np.split[0.7,0.8] means 1st part upto 80%, 2nd part from 80% to 90% , and 3rd part from 90% to full.

training_imagenames=[src+'/'+ name for name in training_imagenames.tolist()] # here we are creating 3 sublists training_imagenames,validation_imagenames,test_imagenames.

validation_imagenames=[src+'/'+ name for name in validation_imagenames.tolist()]

test_imagenames=[src+'/'+ name for name in test_imagenames.tolist()]

print('Total images:',len(allImageNames))
print('Training:',len(training_imagenames))
print('Validation:',len(validation_imagenames))
print('Testing:',len(test_imagenames))

#Now copying into train,test and validation folders

for name in training_imagenames:
    shutil.copy(name,'Train/'+ currentCls)
for name in validation_imagenames:
    shutil.copy(name,'Validation/'+ currentCls)
for name in test_imagenames:
    shutil.copy(name,'Test/'+ currentCls)






i=0

directory = 'Train/closednew'
for file in os.listdir(directory):
  img_path = os.path.join(directory, file)
  img = cv2.imread(img_path,0)
  print(img_path)
  #cv2_imshow(img)
  #horizontal_img = cv2.flip( img, 1 ) 
  img = cv2.resize(img, (50, 50))
  #cv2_imshow(img_data_resized)
  #img = cv2.imread(img,0)
  equ = cv2.equalizeHist(img)
  #res = np.hstack((img,equ)) #stacking images side-by-side
  #cv2.imwrite('res.png',res)
  #print("2")
  #cv2_imshow(res)
  i=i+1
  print(i)



  cv2.imwrite('Train1/closednew/' +  file + '_hist.jpg', equ) 
print(i)









directory = 'Train/opennew'
i=0
for file in os.listdir(directory):
  img_path = os.path.join(directory, file)
  img = cv2.imread(img_path,0)
  #print("1")
  #cv2_imshow(img)
  #horizontal_img = cv2.flip( img, 1 ) 
  img = cv2.resize(img, (50, 50))
  #cv2_imshow(img_data_resized)
  #img = cv2.imread(img,0)
  equ = cv2.equalizeHist(img)
  res = np.hstack((img,equ)) #stacking images side-by-side
  #cv2.imwrite('res.png',res)
  print("2")
  #cv2_imshow(res)
  i=i+1



  cv2.imwrite('Train1/opennew/' +  file + '_hist.jpg', equ) 
print(i)





directory = 'Validation/closednew'
for file in os.listdir(directory):
  img_path = os.path.join(directory, file)
  img = cv2.imread(img_path,0)
  #print("1")
  #cv2_imshow(img)
  #horizontal_img = cv2.flip( img, 1 ) 
  img = cv2.resize(img, (50, 50))
  #cv2_imshow(img_data_resized)
  #img = cv2.imread(img,0)
  equ = cv2.equalizeHist(img)
  res = np.hstack((img,equ)) #stacking images side-by-side
  #cv2.imwrite('res.png',res)
  print("2")
 # cv2_imshow(res)




  cv2.imwrite('Validation1/closednew/' +  file + '_hist.jpg', equ) 




directory = 'Test/closednew'
for file in os.listdir(directory):
  img_path = os.path.join(directory, file)
  img = cv2.imread(img_path,0)
  #print("1")
  #cv2_imshow(img)
  #horizontal_img = cv2.flip( img, 1 ) 
  img = cv2.resize(img, (50, 50))
  #cv2_imshow(img_data_resized)
  #img = cv2.imread(img,0)
  equ = cv2.equalizeHist(img)
  res = np.hstack((img,equ)) #stacking images side-by-side
  #cv2.imwrite('res.png',res)
  print("2")
  #cv2_imshow(res)




  cv2.imwrite('Test1/closednew/' +  file + '_hist.jpg', equ) 




i=0
directory = 'Validation/opennew'
for file in os.listdir(directory):
  img_path = os.path.join(directory, file)
  img = cv2.imread(img_path,0)
  #print("1")
  #cv2_imshow(img)
  #horizontal_img = cv2.flip( img, 1 ) 
  img = cv2.resize(img, (50, 50))
  #cv2_imshow(img_data_resized)
  #img = cv2.imread(img,0)
  equ = cv2.equalizeHist(img)
  res = np.hstack((img,equ)) #stacking images side-by-side
  #cv2.imwrite('res.png',res)
  print("2")
  #cv2_imshow(res)
  i=i+1




  cv2.imwrite('Validation1/opennew/' +  file + '_hist.jpg', equ) 
print(i)




i=0
directory = 'Test/opennew'
for file in os.listdir(directory):
  img_path = os.path.join(directory, file)
  img = cv2.imread(img_path,0)
  #print("1")
  #cv2_imshow(img)
  #horizontal_img = cv2.flip( img, 1 ) 
  img = cv2.resize(img, (50, 50))
  #cv2_imshow(img_data_resized)
  #img = cv2.imread(img,0)
  equ = cv2.equalizeHist(img)
  res = np.hstack((img,equ)) #stacking images side-by-side
  #cv2.imwrite('res.png',res)
  print("2")
  #cv2_imshow(res)
  i=i+1




  cv2.imwrite('Test1/opennew/' +  file + '_hist.jpg', equ)
print(i) 












      






