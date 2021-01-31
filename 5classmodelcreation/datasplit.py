import os
import numpy as np
import shutil



centreposCls = '/resizedcentreimages' #centreposCls is centre images
leftposCls = '/resizedleftimages' #leftposCls is left images
rightposCls = '/resizedrightimages' #rightposCls is right images
upposCls = '/up'
downposCls = '/Down'


os.makedirs('Train'+ centreposCls)
os.makedirs('Train'+ leftposCls)
os.makedirs('Train'+ rightposCls)
os.makedirs('Train'+ upposCls)
os.makedirs('Train'+ downposCls)
os.makedirs('Validation'+ centreposCls)
os.makedirs('Validation'+ leftposCls)
os.makedirs('Validation'+ rightposCls)
os.makedirs('Validation'+ upposCls)
os.makedirs('Validation'+ downposCls)
os.makedirs('Test'+ centreposCls)
os.makedirs('Test'+ leftposCls)
os.makedirs('Test'+ rightposCls)
os.makedirs('Test'+ upposCls)
os.makedirs('Test'+ downposCls)




currentCls=centreposCls
src='FINAL2/'+currentCls #here source becomes inside cell_images,the currentCls ie posCls.
allImageNames=os.listdir(src) # here all the image names will be inside a list named allImageNames.
np.random.shuffle(allImageNames) #shuffle the list containing all the images.
training_imagenames,validation_imagenames,test_imagenames=np.split(np.array(allImageNames),[int(len(allImageNames)*0.8),int(len(allImageNames)*0.9)])
#in the above code we splitted the whole parasitized set ie allImageNames into 3 for training ,testing,validation.  np.split[0.7,0.8] means 1st part upto 70%, 2nd part from 70% to 80% , and 3rd part from 80% to full.

training_imagenames=[src+'/'+ name for name in training_imagenames.tolist()] # here we are creating 3 sublists training_imagenames,validation_imagenames,test_imagenames.

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
src='FINAL2/'+currentCls #here source becomes inside cell_images,the currentCls ie posCls.
allImageNames=os.listdir(src) # here all the image names will be inside a list named allImageNames.
np.random.shuffle(allImageNames) #shuffle the list containing all the images.
training_imagenames,validation_imagenames,test_imagenames=np.split(np.array(allImageNames),[int(len(allImageNames)*0.8),int(len(allImageNames)*0.9)])
#in the above code we splitted the whole parasitized set ie allImageNames into 3 for training ,testing,validation.  np.split[0.7,0.8] means 1st part upto 70%, 2nd part from 70% to 80% , and 3rd part from 80% to full.

training_imagenames=[src+'/'+ name for name in training_imagenames.tolist()] # here we are creating 3 sublists training_imagenames,validation_imagenames,test_imagenames.

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
src='FINAL2/'+currentCls #here source becomes inside cell_images,the currentCls ie posCls.
allImageNames=os.listdir(src) # here all the image names will be inside a list named allImageNames.
np.random.shuffle(allImageNames) #shuffle the list containing all the images.
training_imagenames,validation_imagenames,test_imagenames=np.split(np.array(allImageNames),[int(len(allImageNames)*0.8),int(len(allImageNames)*0.9)])
#in the above code we splitted the whole parasitized set ie allImageNames into 3 for training ,testing,validation.  np.split[0.7,0.8] means 1st part upto 70%, 2nd part from 70% to 80% , and 3rd part from 80% to full.

training_imagenames=[src+'/'+ name for name in training_imagenames.tolist()] # here we are creating 3 sublists training_imagenames,validation_imagenames,test_imagenames.

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






currentCls=upposCls
src='FINAL2/'+currentCls #here source becomes inside cell_images,the currentCls ie posCls.
allImageNames=os.listdir(src) # here all the image names will be inside a list named allImageNames.
np.random.shuffle(allImageNames) #shuffle the list containing all the images.
training_imagenames,validation_imagenames,test_imagenames=np.split(np.array(allImageNames),[int(len(allImageNames)*0.8),int(len(allImageNames)*0.9)])
#in the above code we splitted the whole parasitized set ie allImageNames into 3 for training ,testing,validation.  np.split[0.7,0.8] means 1st part upto 70%, 2nd part from 70% to 80% , and 3rd part from 80% to full.

training_imagenames=[src+'/'+ name for name in training_imagenames.tolist()] # here we are creating 3 sublists training_imagenames,validation_imagenames,test_imagenames.

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





currentCls=downposCls
src='FINAL2/'+currentCls #here source becomes inside cell_images,the currentCls ie posCls.
allImageNames=os.listdir(src) # here all the image names will be inside a list named allImageNames.
np.random.shuffle(allImageNames) #shuffle the list containing all the images.
training_imagenames,validation_imagenames,test_imagenames=np.split(np.array(allImageNames),[int(len(allImageNames)*0.8),int(len(allImageNames)*0.9)])
#in the above code we splitted the whole parasitized set ie allImageNames into 3 for training ,testing,validation.  np.split[0.7,0.8] means 1st part upto 70%, 2nd part from 70% to 80% , and 3rd part from 80% to full.

training_imagenames=[src+'/'+ name for name in training_imagenames.tolist()] # here we are creating 3 sublists training_imagenames,validation_imagenames,test_imagenames.

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









































