from joblib import load
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
from mtcnn import MTCNN
from sklearn.model_selection import train_test_split
import glob
import cv2
import numpy as np

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"


detector=MTCNN()
clf=load("gender_classifier")


def face_detector(img):
    try:
        rgb_img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        out=detector.detect_faces(rgb_img)[0]
        x,y,w,h=out["box"]
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
       

        return img[y:y+h,x:x+w],img,x,y
    except:
        pass

for item in (glob.glob("fifth week_ML\gender_detection\\test_gender\\*")):

    img=cv2.imread(item)
    face,detcted_face,x,y=face_detector(img)


    if face is None:
            continue

    face=cv2.resize(face,(32,32))
    face=face.flatten()
    face=face/255.0
    
    out=clf.predict(np.array([face]))[0] 
    cv2.putText(detcted_face,out,(x,y-10),cv2.FONT_HERSHEY_PLAIN,1.2,(0,255,0),2) 



    cv2.imshow("image:",detcted_face) 
    cv2.waitKey(0)

cv2.destroyAllWindows()   
