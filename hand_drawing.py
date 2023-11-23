# TechVidvan hand Gesture Recognizer

# import necessary packages

import math
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from keras.models import load_model

# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Load the gesture recognizer model
model = load_model('mp_hand_gesture')

# # Load class names
# f = open('gesture.names', 'r')
# classNames = f.read().split('\n')
# f.close()
# print(classNames)

#restituisce il verso della mano (palmo--> "PALM" / dorso --> "BACK")
def verso(landmarks, mano):
    PALMO = "PALM"
    DORSO = "BACK"
    #se mano sinistra e coordinata x del punto 5 > coordinata x del punto 17
    if mano.lower()=="left" and landmarks[5][0] > landmarks[17][0]:
        return PALMO
    #se mano destra e coordinata x del punto 5 < coordinata x del punto 17
    if mano.lower()=="right" and landmarks[5][0] < landmarks[17][0]:
        return PALMO    
    return DORSO

#conta il numero di dita
def numeroDita(landmarks, mano, verso):
    numero = 0

    #pollice (dipende dalla mano e dal verso della mano)
    #v1: funziona solo se la mano NON è in pos.orizzontale
    if verso.lower() == "palm":
        if (mano.lower() == "left" and landmarks[4][0] > landmarks[3][0]) or (mano.lower() == "right" and landmarks[4][0] < landmarks[3][0]):
            numero+=1
    elif verso.lower() == "back":
        if (mano.lower() == "left" and landmarks[4][0] < landmarks[3][0]) or (mano.lower() == "right" and landmarks[4][0] > landmarks[3][0]):
            numero+=1

    #indice
    if landmarks[8][1] < landmarks[6][1]:
        numero+=1
    #medio
    if landmarks[12][1] < landmarks[10][1]:
        numero+=1
    #anulare
    if landmarks[16][1] < landmarks[14][1]: 
        numero+=1
    #mignolo
    if landmarks[20][1] < landmarks[18][1]:   
        numero+=1

    return numero

#calcola l'angolo di inclinazione della mano (in gradi)
#restituisce le coordinate del punto medio dei punti 5-9-13-17
#restituisce l'angolo in gradi     
def angoloInclinazione(landmarks):
    #coordinate punto 0 (centro mano)
    l0X = landmarks[0][0]
    l0Y = landmarks[0][1]
    #calcolo le coordinate del punto medio 5-9-13-17
    mediaX = (landmarks[5][0] + landmarks[9][0] + landmarks[13][0] + landmarks[17][0]) / 4
    mediaY = (landmarks[5][1] + landmarks[9][1] + landmarks[13][1] + landmarks[17][1]) / 4
    #delta coodinate punto 0 e punto medio 5-9-13-17 
    deltaY = mediaY - l0Y
    deltaX = mediaX - l0X
    #correggo lo 0 al denominatore
    if deltaX == 0:
        deltaX = 0.000001
    #calcolo il coeff.angolare della retta che unisce il palmo con il punto medio 5-9-13-17
    coeffAng = (deltaY) / (deltaX)
    #converto il coeff.angolare in gradi
    if l0X > mediaX and l0Y > mediaY:
        gradi = math.atan(coeffAng) * 180 / math.pi
    elif l0X < mediaX and l0Y > mediaY:
        gradi = 180-abs(math.atan(coeffAng) * 180 / math.pi)
    elif l0X < mediaX and l0Y < mediaY:
        gradi = 180+math.atan(coeffAng) * 180 / math.pi
    else:
        gradi = 360-abs(math.atan(coeffAng) * 180 / math.pi)
    #restituisco le coordinate del punto medio dei punti 5-9-13-17
    #restituisco l'angolo in gradi  
    return int(mediaX), int(mediaY), gradi


# Initialize the webcam
cap = cv2.VideoCapture(0)
drawMode = False
old_drawing = []
drawingPoints = []

while True:
    # Read each frame from the webcam
    _, frame = cap.read()

    x  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
    y = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)   # float `height`

    # Flip the frame vertically
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get hand landmark prediction
    result = hands.process(framergb)

    # print(result)
    
    className = ''

    #switch on/off drawing mode
    if cv2.waitKey(1) == ord('d'):
        drawMode = not drawMode
        if not drawMode:
            old_drawing = []
            for point in drawingPoints:
                old_drawing.append(point)
            drawingPoints = []
                    

    #erase screen
    if cv2.waitKey(1) == ord('x'):
        drawMode = False
        drawingPoints = []
        old_drawing = []

    # post process the result
    if result.multi_hand_landmarks:

        landmarks = []
        for handslms in result.multi_hand_landmarks:

            indiceMano = result.multi_hand_landmarks.index(handslms)
            etichettaMano = result.multi_handedness[indiceMano].classification[0].label
            for lm in handslms.landmark:
                # print(id, lm)
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)

                landmarks.append((lmx, lmy))
            if drawMode:
                drawingPoints.append((landmarks[8][0], landmarks[8][1]))


            # Drawing landmarks on frames
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

            #visualizzo sullo schermo
            #verso della mano
            # v = verso(landmarks, etichettaMano)
            # #numero dita visibili
            # nDita = numeroDita(landmarks, etichettaMano, v)
            # #angolo inclinazione
            # xm, ym, gradi = angoloInclinazione(landmarks)
            # cv2.putText(frame, v, [10, 80], cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            # cv2.putText(frame, str(nDita), [10, 40], cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            # cv2.putText(frame, str(gradi), [10, 60], cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    #Disegno partendo dall'indice    
    if drawMode:
        cv2.polylines(frame, [np.array(drawingPoints)], False, (255, 255, 255), 3)
    else:
        cv2.polylines(frame, [np.array(old_drawing)], False, (255, 255, 255), 3)

        
    cv2.putText(frame, "Draw Mode: "+("enabled" if drawMode else "disabled"), [10, 20], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if drawMode else (0, 0, 255), 2)

            

            # # Predict gesture
            # prediction = model.predict([landmarks])
            # # print(prediction)
            # classID = np.argmax(prediction)
            # className = classNames[classID]

    # # show the prediction on the frame
    # cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
    #                1, (0,0,255), 2, cv2.LINE_AA)

    # Show the final output
    cv2.imshow("Output", frame) 

    if cv2.waitKey(1) == ord('q'):
        break
    

# release the webcam and destroy all active windows
cap.release()

cv2.destroyAllWindows()