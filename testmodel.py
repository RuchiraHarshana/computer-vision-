import cv2
video=cv2.VideoCapture(0)
facedetect=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

recognizer=cv2.face.LBPHFaceRecognizer_create()
recognizer.read("Trainer.yml")
name_list=["","Ruchira",]
while True:
    ret,frame=video.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=facedetect.detectMultiScale(gray,1.3,5)
    for(x,y,w,h)in faces:
        serail,conf=recognizer.predict(gray[y:y+h,x:x+w])
        if conf<50:
          cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 225), 1)
          cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 225), 2)
          cv2.rectangle(frame, (x, y-40), (x + w, y ), (50, 50, 225), -1)

          cv2.putText(frame,name_list[serail],(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)



        else:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 225), 1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 225), 2)
            cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 225), -1)

            cv2.putText(frame, "unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Frame",frame)
    k=cv2.waitKey(1)

    if k==ord('a'):
        break

video.release()
cv2.destroyAllWindows()
print("identification done....")