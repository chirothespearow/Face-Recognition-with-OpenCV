import os
import numpy as np
from PIL import Image
import cv2
import pickle

current_id=0
label_ids={}
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
x_train=[]
y_labels=[]

BASE_DIR =os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR,'images')

recognizer=cv2.face.LBPHFaceRecognizer_create()
 
for root,dirs,files in os.walk(image_dir):
	for file in files:
		if file.endswith('png') or file.endswith('jpg'):
			path= os.path.join(root,file)
			label = os.path.basename(os.path.dirname(path)).replace(' ','-').lower()
			#print(path,label)
			if label in label_ids:
				pass
			else:
				label_ids[label]=current_id
				current_id+=1
			id_=label_ids[label]

			#x_train.append(path)
			#y_label.append(label)
			pil_image=Image.open(path).convert('L') #L for gray scale
			size=(550,550)
			final_image=pil_image.resize(size,Image.ANTIALIAS)
			image_array= np.array(final_image,'uint8')
			faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)
			for (x,y,w,h) in faces:
				roi = image_array[y:y+h,x:x+w]
				x_train.append(roi)
				y_labels.append(id_)
#print(y_labels)
print(label_ids)              
#print(x_train)
with open('labels.bin','wb') as f:
	pickle.dump(label_ids,f)

recognizer.train(x_train,np.array(y_labels))
recognizer.save('Trainer.yml')