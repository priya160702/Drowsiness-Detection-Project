# image processing libarary
import cv2
# for accessing operating system
import os
# for machine learning and trained data
from keras.models import load_model
import numpy as np
# Python modules designed for writing video games. It includes computer graphics and sound libraries
from pygame import mixer    #mixer pygame module for loading and playing sounds
import time

mixer.init()
sound = mixer.Sound('alarm.wav')

# haarcascade files location
face = cv2.CascadeClassifier('C://Users//HP//Desktop//Aksh//Extra//New folder//my_practice_one//haar cascade files//haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('C://Users//HP//Desktop//Aksh//Extra//New folder//my_practice_one//haar cascade files//haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('C://Users//HP//Desktop//Aksh//Extra//New folder//my_practice_one//haar cascade files//haarcascade_righteye_2splits.xml')

# list of function here it means to eyes open and close
lbl = ['Close', 'Open']


model = load_model('C://Users//HP//Desktop//Aksh//Extra//New folder//my_practice_one//models//cnncat2.h5')
# getcwd provide access to operate in current working directory
path = os.getcwd()
# for turn on webcam
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count = 0
score = 0
thicc = 2
# here right eye prediction(rpred)
rpred = [99]
# here left eye prediction(lpred)
lpred = [99]

# for reading the video
while True:
    ret, frame = cap.read()
    height, width = frame.shape[:2]
# converting color to gray scale using opencv
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# method use to detect multiple objects in single frame
    faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))

    left_eye = leye.detectMultiScale(gray)
    right_eye = reye.detectMultiScale(gray)

    cv2.rectangle(frame, (0, height - 50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

# for face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)

    for (x, y, w, h) in right_eye:
        r_eye = frame[y:y + h, x:x + w]
        count = count + 1
        r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye, (24, 24))
        # normalize the pixels
        r_eye = r_eye / 255
        r_eye = r_eye.reshape(24, 24, -1)
        r_eye = np.expand_dims(r_eye, axis=0)
        # rpred = model.predict_classes(r_eye)


        predict_x = model.predict(r_eye)
        rpred = np.argmax(predict_x, axis=1)


        if (rpred[0] == 1):
            lbl = 'Open'
        if (rpred[0] == 0):
            lbl = 'Closed'
        break

    for (x, y, w, h) in left_eye:
        l_eye = frame[y:y + h, x:x + w]
        count = count + 1
        l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
        l_eye = cv2.resize(l_eye, (24, 24))
        l_eye = l_eye / 255
        l_eye = l_eye.reshape(24, 24, -1)
        l_eye = np.expand_dims(l_eye, axis=0)
        # lpred = model.predict_classes(l_eye)

        predict_y = model.predict(l_eye)
        lpred = np.argmax(predict_y, axis=1)

        if (lpred[0] == 1):
            lbl = 'Open'
        if (lpred[0] == 0):
            lbl = 'Closed'
        break

    if (rpred[0] == 0 and lpred[0] == 0):
        score = score + 1
        cv2.putText(frame, "Closed", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    # if(rpred[0]==1 or lpred[0]==1):
    else:
        score = score - 1
        cv2.putText(frame, "Open", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    if (score < 0):
        score = 0
    cv2.putText(frame, 'Score:' + str(score), (100, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    if (score > 12):
        # person is feeling sleepy so we beep the alarm
        cv2.imwrite(os.path.join(path, 'image.jpg'), frame)
        try:
            sound.play()

        except:  # isplaying = False
            pass
        if (thicc < 16):
            thicc = thicc + 2
        else:
            thicc = thicc - 2
            if (thicc < 2):
                thicc = 2
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
