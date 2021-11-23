import cv2 as cv

haar_face = cv.CascadeClassifier("haar_face.xml") # on fait appelle au coefs set de detection  de visages

def rescaleFrame(frame, scale=0.75):
    [height, width] = (frame.shape[0], frame.shape[1])
    dimension = (int(width*scale), int(height*scale))
    return cv.resize(frame, dimension, interpolation=cv.INTER_AREA)

