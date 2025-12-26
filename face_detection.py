import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
from mtcnn import MTCNN
import cv2


detector=MTCNN()
img=cv2.imread("images (1).jfif")
r_img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)


out=detector.detect_faces(r_img)[0]
#print(out)
x,y,w,h=out["box"]
#print(x)
cv2.rectangle(img,(x,y),(x+w , y+h),(0,255,216),2)

confidence=out["confidence"]
#print(confidence)
txt="prob:{:.2f}".format(confidence*100)
cv2.putText(img,txt,(x,y-10),cv2.FONT_HERSHEY_PLAIN,1.5,(0,255,0),1)


kp=out["keypoints"]
for key,value in kp.items():
    print(key,value)
    cv2.circle(img,value,5,(0,0,255),-1)
cv2.imshow("image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
