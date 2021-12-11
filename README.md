# Reconnaissance Facial

**Ce projet regroupe un ensemble de modules pour creer un trainset de visages**

**et entrainer le modele de reconnaissance facial integrer par Opencv a le**

**reconnaitre**

## train (dossier) 

**est le dossier des trainsets**

**les visages de chaque personne sont mis dans un dossier a part**

**et le nom du dossier sera l'etiquette des donnes de ce visages**

## create_trainset.py

**possede une fonction ```create``` qui capture un nombre fini d'images depuis**

**la webcam puis les met dans le dossier ```train``` un dossier portant le nom saisie**

## recorder.py

**est le programme qui effectue la detection des visages en temps real**

## setup.bat

**le ficher d'installation des dependances pour windows**

## setup.sh

**le ficher d'installation des dependances pour Linux**

## trainer.py

**effectue l'apprentisage en puissant son trainset depuis un dossier specifier**

**dans notre cas c'est le dossier ```train```**
