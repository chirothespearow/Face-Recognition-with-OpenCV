import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
cap = cv2.VideoCapture(0)
#Getting names of people stored in database
labels={}
with open('labels.bin','rb') as f:
	labels = pickle.load(f)
key_list = list(labels.keys()) 
val_list = list(labels.values()) 
recognizer=cv2.face.LBPHFaceRecognizer_create()
recognizer.read('Trainer.yml')

while True:
	#reading frame
	ret, frame= cap.read()
	#displaying frame
		#creating a grayscale frame
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	#detecting faces
	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
	#x,y = coordinates, w,h= widht and height
	for (x,y,w,h) in faces:
		#print(x,y,w,h)
		roi_gray = gray[y:y+h,x:x+w] #cutting out the face
		id_,conf=recognizer.predict(roi_gray) #searching for face in database!
		if conf>=45 and conf<=85:
			
			print(key_list[val_list.index(id_)])
			font = cv2.FONT_HERSHEY_SIMPLEX
			name = key_list[val_list.index(id_)]
			color = (255,255,255)
			stroke = 2
			cv2.putText(frame,name,(x,y),font,1,color,stroke,cv2.LINE_AA)

		cv2.imwrite('image.png',roi_gray) #saving
		#creating rectangle around face
		color = (255,0,0)
		stroke = 2
		width=x+w
		height=y+h
		cv2.rectangle(frame, (x,y),(width,height),color,stroke)
	cv2.imshow('frame',frame)


	if cv2.waitKey(20) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
	