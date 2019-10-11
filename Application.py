import cv2
import numpy as np
from keras.models import load_model

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

stroke = 2
color = (255, 255, 255)
model = load_model('model_CNN_V2.h5')
font = cv2.FONT_HERSHEY_SIMPLEX
Gender = {0: "Male", 1: "Female"}

def normalizeImage(img):
    IMG_SIZ = 120
    new_img = cv2.resize(img, (IMG_SIZ, IMG_SIZ))
    image = new_img.reshape((120, 120, 1))
    image = image.astype('float32') / 255
    return image


while(True):

    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame_gray, 1.2, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = frame_gray[y:y + h, x:x + w]     # region of interest
        roi_color = frame[y:y + h, x:x + w]
        img = np.array(roi_gray)
        img = normalizeImage(img)
        prediction = model.predict([[img]]).argmax()
        gender = Gender.get(prediction)
        cv2.putText(frame, gender, (x, y), font, 1, color, stroke, cv2.LINE_AA)

    cv2.imshow('Gender Prediction', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()