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

path48x48 = "./StockImagesOpenCV/48x48"
pathNormal = "./StockImagesOpenCV/normal"

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

img_counter = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        for element in os.listdir("./StockImagesOpenCV/48x48"):
            os.remove("./StockImagesOpenCV/48x48/" + element)
        break
    elif k%256 == 32:
        # SPACE pressed
        
        #Noms fichiers
        img_name = "opencv_frame_{}.png".format(img_counter)
        img_name_crop = "opencv_frame_{}_crop.png".format(img_counter)

        #Traitement photo
        img = frame
        imgCam = frame
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            #Enregistrement fichier crop et en B&W + 48*48 pixels
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
            print(tabEmotion[pred[0]])

            
            #Enregistrement fichier normal

            cv2.rectangle(imgCam, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(imgCam, tabEmotion[pred[0]], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
            cv2.imwrite(os.path.join(pathNormal , img_name), imgCam)

        
        print("{} written!".format(img_name))
            

        #Incrémentation pour nommage fichier
        img_counter += 1

cam.release()

cv2.destroyAllWindows()

