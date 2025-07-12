import numpy as np
from PIL import Image
import os 
import cv2
import time

def Draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_image, scaleFactor, minNeighbors)

    coords = []
    for (x, y, w, h) in features:
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        id, pred = clf.predict(gray_image[y:y+h, x:x+w])
        confidence = int(100 * (1 - pred / 300))

        if confidence > 85:
            if id == 1:
                cv2.putText(img, " user1 ", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
            elif id == 2:
                cv2.putText(img, "Manu", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        else:
            cv2.putText(img, "unknown", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)

        coords = [x, y, w, h]
    return coords

def recognize(img, clf, faceCascade):
    Draw_boundary(img, faceCascade, 1.1, 10, (255, 255, 255), "Face", clf)
    return img

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")    
clf = cv2.face.LBPHFaceRecognizer_create()
clf.read("classifier.xml")

video_capture = cv2.VideoCapture(0)

start_time = time.time()
max_duration = 120  # seconds (2 minutes timeout)

while True:
    ret, img = video_capture.read()
    if not ret:
        print("Failed to capture image")
        break

    img = recognize(img, clf, faceCascade)
    cv2.imshow("Face detection", img)

    key = cv2.waitKey(1)

    # Exit conditions
    if cv2.getWindowProperty("Face detection", cv2.WND_PROP_VISIBLE) < 1:
        print("Exit: Window closed by user")
        break
    elif time.time() - start_time > max_duration:
        print("Exit: Auto timeout after 2 minutes")
        break

video_capture.release()
cv2.destroyAllWindows()
print("Camera and windows released.")
