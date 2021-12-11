import cv2 as cv
import create_trainset as cd
import trainer
from tool_img import rescaleFrame, haar_face

cd.create()
trainDataFile = "entrainement.yml"
peopleSet = trainer.train("train", trainDataFile)

faceRecognizer = cv.face.LBPHFaceRecognizer_create()
faceRecognizer.read(trainDataFile)

img = cv.imread("/home/talibe/Bureau/opencv/train/jeff_bezos/jeff1.jpeg")

img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img = rescaleFrame(img, 0.8)


face_rect = haar_face.detectMultiScale(img, 1.1, 1)

for (x, y, w, h) in face_rect:
    face = img[x:x+w, y:y+h]
    label, confidence = faceRecognizer.predict(face)
    #print("label predit : ", label)
    print(f'label = {peopleSet[label]} with confidence of {confidence}')

    cv.putText(img,peopleSet[label], (20, 20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), thickness=2)
    cv.imshow("img", img)

cv.waitKey(0)
