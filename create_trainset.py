import cv2 as cv
import os
from time import sleep


def create():
    nom = input("entrer le votre nom")
    os.mkdir(f"train/{nom}")

    capture = cv.VideoCapture(1)
    i = 0
    while True:
        isTrue, image = capture.read() #prendre un capture
        cv.imwrite("train/{nom}/cap_{str(i)}.jpg", image) # enregistrer la capture
        i += 1
        sleep(0.75)

if __name__ == '__main__':
    create()