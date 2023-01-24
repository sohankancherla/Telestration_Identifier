import io, os
import cv2
from google.cloud import vision 
from google.cloud import vision_v1
from os import listdir
from bs4 import BeautifulSoup
import csv
import collections
import math
import pandas as pd
# from google.cloud import types

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'api-key.json'
client = vision.ImageAnnotatorClient()

def detect_handwrite(path):
    bounding_box = []

    imageforcv = cv2.imread(path)
    """Detects hand writting or text features in an image."""

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.document_text_detection(image=image)
    texts = response.text_annotations #this will output an array of all the words' objects that contain bounding box info
    texts = texts[1:] # get rid of the first bounding box which is a bounding box that cover the area of all existed word
    
    for text in texts:
        vertices = ([ [vertex.x, vertex.y] for vertex in text.bounding_poly.vertices]) 
        box1 = vertices[0] + vertices[2] #[[xmin,ymin, xmax, ymax]] where [xmin,ymin]left-up corner,  [xmax,ymax]right-bot corner a bounding box
        bounding_box.append(box1)
    return bounding_box


# folder_dir = r"GT test/images/" #your images folder dir
# for images in os.listdir(folder_dir):
#     bounding_box = [] # empty the array for next images
#
#     if (images.endswith(".png") or images.endswith(".jpg") or images.endswith(".jpeg") or images.endswith(".ppm")  or images.endswith(".webp")):
#         print(images)
#
#         # if the picture doesnt contain any detectable text by Google it will go to next image automatically
#         detect_handwrite(folder_dir +images)
#     print(bounding_box)



