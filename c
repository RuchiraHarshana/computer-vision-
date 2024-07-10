import cv2
import numpy as np
from PIL import Image
import os
recognizer=cv2.face.LBPHFaceRecognizer_create()



path = "datasets"


def get_image_id(path):
    imagepaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    ids = []

    for imagepath in imagepaths:
        faceImage = Image.open(imagepath).convert('L')
        facenp = np.array(faceImage)
        Id = os.path.splitext(os.path.split(imagepath)[-1])[0].split('.')[1]
        Id=int(Id)
        ids.append(Id)
        faces.append(facenp)
        cv2.imshow("Trining",facenp)
        cv2.waitKey(1)

    return ids, faces


IDs,facedata=get_image_id(path)
recognizer.train(facedata,np.array(IDs))
recognizer.write("Trainer.yml")
cv2.destroyAllWindows()
print("training compleate")