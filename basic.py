import numpy as np
import cv2
import face_recognition

imgElong = face_recognition.load_image_file('ImagesBasic/elon.jpg')
imgElong = cv2.cvtColor(imgElong, cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('ImagesBasic/elonTest.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)


faceLoc = face_recognition.face_locations(imgElong)[0]
encodeElong = face_recognition.face_encodings(imgElong)[0]
cv2.rectangle(imgElong, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)



cv2.imshow('elon', imgElong)
cv2.imshow('elon test', imgTest)
cv2.waitKey(0)
