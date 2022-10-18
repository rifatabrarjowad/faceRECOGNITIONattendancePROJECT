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

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeElongTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)

Results = face_recognition.compare_faces([encodeElong], encodeElongTest)
faceDis = face_recognition.face_distance([encodeElong], encodeElongTest)
print(Results, faceDis)


cv2.imshow('elon', imgElong)
cv2.imshow('elon test', imgTest)
cv2.waitKey(0)
