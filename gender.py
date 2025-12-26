import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from mtcnn import MTCNN
from joblib import dump
import glob


detector=MTCNN()
data=[]
labels=[]

def face_detection(img):
    try:
        rgb_img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        out=detector.detect_faces(rgb_img)[0]
        x,y,w,h=out["box"]
        return  img[y:y+h , x:x+w]
    
    except:
        pass
 
for i,item in enumerate(glob.glob("fifth week_ML\gender_detection\Gender\\*\\*")):
    img=cv2.imread(item)
    face=face_detection(img)
    if face is None:
        continue
    face=cv2.resize(face,(32,32))
    face=face.flatten()
    face=face/255.0
    #print(face)
    data.append(face)
    label=item.split("\\")[-2]
    labels.append(label)

    if i % 100 ==0 :
        print("[INFO]:{}/3300 processed".format(i))

data=np.array(data)

x_train,x_test,y_train,y_test=train_test_split(data,labels,test_size=0.2)
clf=SGDClassifier()
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)

acc=accuracy_score(y_test,y_pred)
print("accuracy:{:.2f}".format(acc*100))
dump(clf,"gender_classifier")
