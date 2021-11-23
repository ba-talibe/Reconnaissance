import cv2 as cv
import numpy as np
from tool_img import rescaleFrame, haar_face
import trainer
import json




capture = cv.VideoCapture(0) # on demare la camera


faceRecognizer = cv.face.LBPHFaceRecognizer_create()
faceRecognizer.read("entrainement.yml")
with open("persons.json", 'r') as file:
    persons = json.load(file)

while True:
    isTrue, image = capture.read()

    image_gris = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    face_rect = haar_face.detectMultiScale(image_gris, 1.1, 4)

    for (x, y, w, h) in face_rect:
        face = image_gris[x:x+w, y:y+h]
        try:
            label, confidence = faceRecognizer.predict(face)
        except Exception as e:
            print(e)
        #print("label predit : ", label)
        print(f'label = {persons[str(label)]} with confidence of {confidence}')

        cv.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        if confidence < 150:
            cv.putText(image,persons[str(label)], (x, y+h), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), thickness=2)

    cv.imshow("video", image)
        

    if cv.waitKey(20) and 0xFF==ord('c'):
        break
