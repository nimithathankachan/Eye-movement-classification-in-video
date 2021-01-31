#********************left right centre and blink detection******************

import dlib
import numpy as np
import cv2
import tensorflow as tf 

from keras.models import load_model
cap=cv2.VideoCapture(0)

detector=dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

image_array = []


i=0
mem_counter=0
blink=0


lenet = load_model('projectmodel2.h5')
blinklenet = load_model('blinkhist.hdf5')

def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)
while True:
    
    _, frame = cap.read()
    i=i+1;
    frame=cv2.flip(frame,1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = detector(gray)
    
    if(len(faces)==0):
        print("no face");
        cv2.putText(frame, "no faces", (10, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow('left right centre', frame) 
              
    for face in faces:
        print(i,"frame no")
        x, y = face.left(), face.top()
        x1, y1 = face.right(), face.bottom()
        cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)
        landmarks = predictor(gray, face)
        
        
        left_eye_region = np.array([(landmarks.part(36).x, landmarks.part(36).y),
                            (landmarks.part(37).x, landmarks.part(37).y),
                            (landmarks.part(38).x, landmarks.part(38).y),
                            (landmarks.part(39).x, landmarks.part(39).y),
                            (landmarks.part(40).x, landmarks.part(40).y),
                            (landmarks.part(41).x, landmarks.part(41).y)], np.int32)
        min_x = np.min(left_eye_region[:, 0])
        max_x = np.max(left_eye_region[:, 0])
        min_y = np.min(left_eye_region[:, 1])
        max_y = np.max(left_eye_region[:, 1])
        
        eye = frame[min_y: max_y, min_x: max_x]
        
        
        
        eye_resized = cv2.resize(eye, (50, 50)) #make image 50*50 size
        
        cv2.imwrite('Framesleft/'+str(1)+'.jpg',eye_resized)
        
        
        img = cv2.imread('Framesleft/'+str(1)+'.jpg',0)
   
        img = cv2.resize(img, (50, 50))
  
        equ = cv2.equalizeHist(img)
        res = np.hstack((img,equ)) #stacking images side-by-side
  
        
        cv2.imwrite('Framesleft1/' + str(1)+ '_hist.jpg', equ) 
        equ = cv2.imread('Framesleft1/' + str(1)+ '_hist.jpg',cv2.IMREAD_COLOR)
        
        equ = cv2.resize(equ, (50, 50)) 
        img_data_reshaped = equ.reshape((1, 50, 50,3)) 
        


        right_eye_region = np.array([(landmarks.part(42).x, landmarks.part(42).y),
                            (landmarks.part(43).x, landmarks.part(43).y),
                            (landmarks.part(44).x, landmarks.part(44).y),
                            (landmarks.part(45).x, landmarks.part(45).y),
                            (landmarks.part(46).x, landmarks.part(46).y),
                            (landmarks.part(47).x, landmarks.part(47).y)], np.int32)
        min_x = np.min(right_eye_region[:, 0])
        max_x = np.max(right_eye_region[:, 0])
        min_y = np.min(right_eye_region[:, 1])
        max_y = np.max(right_eye_region[:, 1])
        eyeright = frame[min_y: max_y, min_x: max_x]

        
        eyer_resized = cv2.resize(eyeright, (50, 50)) #make image 50*50 size
        
        cv2.imwrite('Framesright/'+str(1)+'r.jpg',eyer_resized)
        
        
        imgr = cv2.imread('Framesright/'+str(1)+'r.jpg',0)
  
        imgr = cv2.resize(imgr, (50, 50))
  
        equr = cv2.equalizeHist(imgr)
        
        cv2.imwrite('Framesright1/' + str(1)+ 'r_hist.jpg', equr) 
        equr = cv2.imread('Framesright1/' + str(1)+ 'r_hist.jpg',cv2.IMREAD_COLOR)
        
        equr = cv2.resize(equr, (50, 50)) 
        imgr_data_reshaped = equr.reshape((1, 50, 50,3)) 

        blinkresultl=blinklenet.predict(img_data_reshaped)
        blinkresultr=blinklenet.predict(imgr_data_reshaped)
        
        blinkl=[]
        blinkr=[]
  
        blinkl=blinkresultl.tolist()
        blinkr=blinkresultr.tolist()
        print("blinkl",blinkl)
        print("blinkr",blinkr)
        bl=int(blinkl[0][0])
        br=int(blinkr[0][0])
        print(bl,"bl")
        print(br,"br")
        
        b=(bl+br)/2
        print(b,"n")

        



        if(b<=0.5):
        	mem_counter=mem_counter+1
        	cv2.putText(frame, "CLOSE", (10, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 6)
		
        else:
        	cv2.putText(frame, "OPEN", (10, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 6)
        	if(2<=mem_counter<=16):
        		blink=blink+1
        	mem_counter=0
     

        	result=lenet.predict(img_data_reshaped)
  
        	l=[]
  
        	l=result.tolist()
        	print(l,"listl")
        	nl=(l[0][0])
  
        	ml=(l[0][1])
        	ol=(l[0][2])
        	
        	print("nl,ml,ol",nl,ml,ol)

        	resultr=lenet.predict(imgr_data_reshaped)
  
        	lr=[]
  #print(result)
        	lr=resultr.tolist()
        	print(lr,"listr")
        	nr=(l[0][0])
  #print(n)
        	mr=(l[0][1])
        	orr=(l[0][2])
        	
        	print(nr,mr,orr,"nr,mr,orr")

        	n=(nl+nr)/2
        	m=(ml+mr)/2
        	o=(ol+orr)/2
        	
        	print(n,m,o,"n,m,o")



        	
        	 
        	
		

        
        	if(n>m and n>o ):
        		cv2.putText(frame, "CENTRE", (10, 80),
				cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 6)
        	elif(m>n and m>o  ):
        		cv2.putText(frame, "LEFT", (10, 80),
				cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 6)
        	else:
        		cv2.putText(frame, "RIGHT", (10, 80),
				cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 6)
        	







                
        	
              
                
                
        
        
        cv2.putText(frame, "Blinks: {}".format(blink), (10, 120),
			cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 6)
        
        frame = cv2.resize(frame, (1700,1000))
        cv2.imshow('left right centre', frame)
        
        cv2.imwrite('Frames/' +  str(i) + '.jpg', frame)     
        
 
        


        
        


    key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
    if key == ord('q'):
        break



cap.release()
cv2.destroyAllWindows()
