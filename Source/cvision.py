from keras.applications import MobileNetV2
from keras.preprocessing import image
from keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import numpy as np
import cv2
mode1=MobileNetV2(weights='imagenet')
def inference(x):
    x=np.expand_dims(x, axis=0)
    x=preprocess_input(x)
    preds=mode1.predict(x)
    return decode_predictions(preds,top=1)[0][0][1]
cap=cv2.VideoCapture(0)
while True:
    ret, frame=cap.read()
    frame=cv2.resize(frame,(224,224))
    predicted=inference(frame[...,::-1])
    cv2.putText(frame, predicted, (5 , 30), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),lineType=cv2.LINE_AA)
    frameout=cv2.resize(frame,(960,600))
    cv2.imshow('Webcam', frameout)
    if cv2.waitKey(1) == 13:
        break
cap.release()
cv2.destroyAllWindows()
