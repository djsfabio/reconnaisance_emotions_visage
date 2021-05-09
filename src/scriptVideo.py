import cv2
import os

import pandas as pd
import numpy as np
import cv2
import tensorflow as tf

from tensorflow import keras

from PIL import Image

model = keras.models.load_model("model63acc.h5")

tabEmotion = ["angry" ,"disgust" ,"fear" ,"happy" ,"sad" ,"surprise" ,"neutral" ]

cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

img_counter = 0

path48x48 = "./StockImagesOpenCV/48x48"
pathNormal = "./StockImagesOpenCV/normal"

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# To capture video from webcam. 
cap = cv2.VideoCapture(0)
# To use a video file as input 
# cap = cv2.VideoCapture('filename.mp4')

while True:
    # Read the frame
    _, img = cap.read()

    _, imgCam = cap.read()

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw the rectangle around each face
    img_name = "opencv_frame_{}.png".format(img_counter)
    img_name_crop = "opencv_frame_{}_crop.png".format(img_counter)
    for (x, y, w, h) in faces:
        crop_img = img[y:y+h, x:x+w]
        crop_img = cv2.resize(crop_img, (48,48))
        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join(path48x48 , img_name_crop), crop_img) 

        dfImage = pd.DataFrame(np.array([0 for i in range(2304)]))
        dfImage = dfImage.T
        img48x48 = Image.open('./StockImagesOpenCV/48x48/'+img_name_crop)
        img48x48 = pd.DataFrame(np.array(img48x48.getdata())).T
            
        img48x48 = img48x48.values.reshape(len(img48x48),48,48,1)

        pred = model.predict(img48x48)
        pred = np.argmax(pred,axis=1)

        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(img, tabEmotion[pred[0]], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
        

    img_counter += 1

    # Display
    cv2.imshow('img', img)

    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        for element in os.listdir("./StockImagesOpenCV/48x48"):
            os.remove("./StockImagesOpenCV/48x48/" + element)
        break
        
# Release the VideoCapture object
cap.release()