# packages required to be installed: opencv-python, numpy, imutils, beautifulsoup4
from imutils.object_detection import non_max_suppression
import numpy as np
import cv2
import os
from bs4 import BeautifulSoup
import glob
import GoogleOCR

# path to images to test
rootdir = "telestration focus on text only/images/"
list_of_files = sorted(filter(os.path.isfile, glob.glob(rootdir + '*')))

boxesWithOverlap = 0
falsePositiveBoxes = 0
faseNegativeBoxes = 0

def is_overlap(boxa, boxb, thresh=0.9):
    anb = {}
    if boxa['r'] < boxb['l'] or boxa['l'] > boxb['r']:
        return False
    if boxa['t'] > boxb['b'] or boxa['b'] < boxb['t']:
        return False
    anb['l'] = max(boxa['l'], boxb['l'])
    anb['r'] = min(boxa['r'], boxb['r'])
    anb['t'] = max(boxa['t'], boxb['t'])
    anb['b'] = min(boxa['b'], boxb['b'])
    I = (anb['r']-anb['l'])*(anb['b']-anb['t'])
    U = boxa['w']*boxa['h']+boxb['w']*boxb['h'] - I
    IOU = I/U
    if IOU < thresh:
        return False
    return True

for fileName in list_of_files:
    image = cv2.imread(fileName)

    xmlFileName = fileName.rsplit('.', maxsplit=1)[0].rsplit('/', maxsplit=1)[1]

    with open('telestration focus on text only/annotations/' + xmlFileName + '.xml', 'r') as f:
        xmlData = f.read()

    soup = BeautifulSoup(xmlData, 'xml')
    xMins = [int(x.text) for x in soup.find_all('xmin')]
    yMins = [int(x.text) for x in soup.find_all('ymin')]
    xMaxs = [int(x.text) for x in soup.find_all('xmax')]
    yMaxs = [int(x.text) for x in soup.find_all('ymax')]

    print(fileName)

    googleBoxes = []
    gtBoxes = []

    googleResponse = GoogleOCR.detect_handwrite(fileName)

    for (startX, startY, endX, endY) in googleResponse:
        googleBoxes.append((startX, startY, endX, endY))

    for (xMin, yMin, xMax, yMax) in zip(xMins, yMins, xMaxs, yMaxs):
        gtBoxes.append((xMin, yMin, xMax, yMax))

    # loop over the OCR bounding boxes
    for (startX, startY, endX, endY) in googleResponse:
        # scale the bounding box coordinates based on the ratios
        startXScale = startX
        startYScale = startY
        endXScale = endX
        endYScale = endY

        # loop over the GT bounding boxes
        for (xMin, yMin, xMax, yMax) in zip(xMins, yMins, xMaxs, yMaxs):
            boxa = {'r': endXScale, 'b': endYScale, 'l': startXScale, 't': startYScale, 'w': endXScale - startXScale,
                    'h': endYScale - startYScale}

            boxb = {'r': xMax, 'b': yMax, 'l': xMin, 't': yMin, 'w': xMax - xMin, 'h': yMax - yMin}

            if is_overlap(boxa, boxb):
                boxesWithOverlap += 1
                try:
                    gtBoxes.remove((xMin, yMin, xMax, yMax))
                    googleBoxes.remove((startX, startY, endX, endY))
                except ValueError:
                    nothing = 0

    falsePositiveBoxes += len(googleBoxes)
    faseNegativeBoxes += len(gtBoxes)

print("Total True Positive: ", boxesWithOverlap)
print("Total False Positive: ", falsePositiveBoxes)
print("Total False Negative: ", faseNegativeBoxes)
