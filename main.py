import cv2
# import streamlit as st
from datetime import datetime


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                     'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                    'haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                      'haarcascade_smile.xml')
nose_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                      'haarcascade_mcs_nose.xml')
# st.title("Motion Detector")
# start = st.button("Start Camera")


# if start:
    # streamlit_image = st.image([])
camera = cv2.VideoCapture(0)
while True:
    check, frame = camera.read()
    eyes_detected = False
    mouth_detected = False
    nose_detected = False
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5)
        roi_gray = gray[y:y + w, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
        mouth = mouth_cascade.detectMultiScale(roi_gray, 1.7, 11)
        nose = nose_cascade.detectMultiScale(gray, 1.7, 11)
        if len(eyes) > 0:
            eyes_detected = True
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh),
                                (0, 255, 0), 5)

        if len(mouth) > 0:
            mouth_detected = True
            for (mx, my, mw, mh) in mouth:
                cv2.rectangle(roi_color, (mx, my), (mx + mw, my + mh),
                            (0, 255, 0), 5)
        if len(nose) > 0:
            nose_detected=True
            for (x,y,w,h) in nose:
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)
                break


    cv2.imshow("test",frame)
    print("eyes: ",eyes_detected)
    print("mouth: ",mouth_detected)
    print("nose: ",nose_detected)
    if cv2.waitKey(1) == ord('q'):
        break

    # streamlit_image.image(frame)
