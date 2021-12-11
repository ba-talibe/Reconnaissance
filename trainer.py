import cv2 as cv
import os, os.path as path
from tool_img import rescaleFrame, haar_face
import numpy as np
import json


def train(train_path, outputfile):

    features = []
    labels = []
    global persons
    persons = os.listdir(train_path)
    pdict = dict()
    for i in range(len(persons)):
        pdict.update({i:persons[i]})
    
    with open("persons.json", "a+") as file:
        file.seek(0)
        file.truncate()
        json.dump(pdict, file)
    
    for person in persons:
        personpath = path.join(train_path, person) #chemin des donnees de la personne
        label = persons.index(person)  #label de la personne qui sera l'index de la personne sur la liste
        
        for img in os.listdir(personpath):
            img_path = path.join(personpath, img) # chemin d el'image

            img_array = cv.imread(img_path) #lire l'image sous forme d'une matrice a trois dimension matrixe 
            img_array = rescaleFrame(img_array, scale=0.8) #on retrecie l'image sans alterer des pixels 
            img_gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY) # on le ramene en format noir et blanc en une matrice a 2 dimension


            rect_face = haar_face.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=4) # on extrait les rectangle des visages
            for (x, y, w, h) in rect_face:
                #un rectangle est caracteriser par sa position et ses dimension par rapport a l'image
                face = img_gray[x:x+w, y:y+h] # on extrait les visages
                features.append(face)
                labels.append(label)

    features = np.array(features, dtype=object)
    labels = np.array(labels)

    facereconizer = cv.face.LBPHFaceRecognizer_create() #on fait appel a notre modele de reconnaissance de visage
    try:
        facereconizer.train(features, labels) # on entraine a reconnaitre notre visage et a les classer selon les labels
    except Exception as e:
        print(e)
    # on sauvegarde les donnnees l'apprentissage
    facereconizer.save(outputfile)
    return persons


if __name__ == '__main__':
    dossier = input("entrer le repertoire d'entrainement : ")
    if not path.isabs(dossier):
        dossier = path.join(os.getcwd(), dossier)
    file = input("entrer le nom du fichier yml de sortie : ")

    train(dossier, file)
    print(persons)