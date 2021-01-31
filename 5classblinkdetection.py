#*************left right centre up down and blink detection*****************
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


lenet = load_model('5class5.hdf5')
blinklenet = load_model('blinkhist.hdf5')

def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)
while True:
    #print("yes")
    _, frame = cap.read()
    i=i+1;
    frame=cv2.flip(frame,1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = detector(gray)
    #print("no",len(faces))
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
        
        
        #print('Original size',eye[0].shape)
        eye_resized = cv2.resize(eye, (50, 50)) #make image 50*50 size
        #print('new size',eye_resized.shape)
        #cv2.imshow('img',eye_resized)
        cv2.imwrite('Framesleft/'+str(1)+'.jpg',eye_resized)
        
        #img_path = os.path.join(directory, file)
        img = cv2.imread('Framesleft/'+str(1)+'.jpg',0)
  #print("1")
  #cv2_imshow(img)
  #horizontal_img = cv2.flip( img, 1 ) 
        img = cv2.resize(img, (50, 50))
  #cv2_imshow(img_data_resized)
  #img = cv2.imread(img,0)
        equ = cv2.equalizeHist(img)
        res = np.hstack((img,equ)) #stacking images side-by-side
  #cv2.imwrite('res.png',res)
  #print("2")
        #cv2.imshow("im",res)
        cv2.imwrite('Framesleft1/' + str(1)+ '_hist.jpg', equ) 
        equ = cv2.imread('Framesleft1/' + str(1)+ '_hist.jpg',cv2.IMREAD_COLOR)
        #print('Framesleft1/' + str(1)+ '_hist.jpg')
        equ = cv2.resize(equ, (50, 50)) 
        img_data_reshaped = equ.reshape((1, 50, 50,3)) #1 image of dimension 50*50 and 3 indicates the no. of channels. ie 3 is for coloured images.
        


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

        #cv2.putText(frame, "eyeright: {}".format(eyeright), (100, 200),
	#		cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) 
        #print("eyeeeee",eyeright)
        #print('Original size',eyeright[0].shape)
        eyer_resized = cv2.resize(eyeright, (50, 50)) #make image 50*50 size
        #print('new size',eyer_resized.shape)
        #cv2.imshow('img',eye_resized)
        cv2.imwrite('Framesright/'+str(1)+'r.jpg',eyer_resized)
        
        #img_path = os.path.join(directory, file)
        imgr = cv2.imread('Framesright/'+str(1)+'r.jpg',0)
  #print("1")
  #cv2_imshow(img)
  #horizontal_img = cv2.flip( img, 1 ) 
        imgr = cv2.resize(imgr, (50, 50))
  #cv2_imshow(img_data_resized)
  #img = cv2.imread(img,0)
        equr = cv2.equalizeHist(imgr)
        #res = np.hstack((img,equ)) #stacking images side-by-side
  #cv2.imwrite('res.png',res)
  #print("2")
        #cv2.imshow("im",res)
        cv2.imwrite('Framesright1/' + str(1)+ 'r_hist.jpg', equr) 
        equr = cv2.imread('Framesright1/' + str(1)+ 'r_hist.jpg',cv2.IMREAD_COLOR)
        #print('Framesright1/' + str(1)+ 'r_hist.jpg')
        equr = cv2.resize(equr, (50, 50)) 
        imgr_data_reshaped = equr.reshape((1, 50, 50,3)) #1 image of dimension 50*50 and 3 indicates the no. of channels. ie 3 is for coloured images.

        blinkresultl=blinklenet.predict(img_data_reshaped)
        blinkresultr=blinklenet.predict(imgr_data_reshaped)
        #print("blinkresultl",blinkresultl)
        #print("blinkresultr",blinkresultr)
        blinkl=[]
        blinkr=[]
  #print(result)
        blinkl=blinkresultl.tolist()
        blinkr=blinkresultr.tolist()
        print("blinkl",blinkl)
        print("blinkr",blinkr)
        bl=int(blinkl[0][0])
        br=int(blinkr[0][0])
        print(bl,"bl")
        print(br,"br")
        #m=int(l[0][1])
        #o=int(l[0][2])
        b=(bl+br)/2
        print(b,"n")

        



        if(b<=0.5):
        	mem_counter=mem_counter+1
        	cv2.putText(frame, "CLOSE", (10, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
		
        else:
        	cv2.putText(frame, "OPEN", (10, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
        	if(2<=mem_counter<=11):
        		blink=blink+1
        	mem_counter=0
     

        	result=lenet.predict(img_data_reshaped)
  #print(result.tolist())
        	l=[]
  #print(result)
        	l=result.tolist()
        	print(l,"listl")
        	nl=(l[0][0])
  #print(n)
        	ml=(l[0][1])
        	ol=(l[0][2])
        	pl=(l[0][3])
        	ql=(l[0][4])
        	print("nl,ml,ol,pl,ql",nl,ml,ol,pl,ql)

        	resultr=lenet.predict(imgr_data_reshaped)
  #print(result.tolist())
        	lr=[]
  #print(result)
        	lr=resultr.tolist()
        	print(lr,"listr")
        	nr=(l[0][0])
  #print(n)
        	mr=(l[0][1])
        	orr=(l[0][2])
        	pr=(l[0][3])
        	qr=(l[0][4])
        	print(nr,mr,orr,pr,qr,"nr,mr,orr,pr,qr")

        	n=(nl+nr)/2
        	m=(ml+mr)/2
        	o=(ol+orr)/2
        	p=(pl+pr)/2
        	q=(ql+qr)/2
        	print(n,m,o,p,q,"n,m,o,p,q")



        	 
        	
		

        #cv2.putText(frame, n, (10, 30),
	#		cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        	if(n>m and n>o and n>p and n>q):
       			cv2.putText(frame, "DOWN", (30, 100),
				cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 6)
        	elif(m>n and m>o and m>p and m>q ):
       			cv2.putText(frame, "CENTRE",(30,100),
				cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 6)
        	elif(o>n and o>m and o>p and o>q ):
        		cv2.putText(frame, "LEFT", (30, 100),
				cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 6)
        	elif(p>n and p>o and p>m and p>q ):
        		cv2.putText(frame, "RIGHT", (30, 100),
				cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 6)
        	else:
        		cv2.putText(frame, "UP", (30, 100),
				cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 6)
        	
              
                
                
        
        
        cv2.putText(frame, "Blinks: {}".format(blink), (300, 400),
			cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 4)
        
        #########cv2.putText(frame, "mem_counter: {}".format(mem_counter), (300, 400),
			###########cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        frame = cv2.resize(frame, (1700,1000))
        cv2.imshow('left right centre', frame)
        
        cv2.imwrite('Frames/' +  str(i) + '.jpg', frame) 
        
        



        
        


    key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
    if key == ord('q'):
        break



cap.release()
cv2.destroyAllWindows()
