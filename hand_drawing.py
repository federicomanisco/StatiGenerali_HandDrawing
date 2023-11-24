# import necessary packages
import math
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from keras.models import load_model



# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

# Load the gesture recognizer model
model = load_model('mp_hand_gesture')

# Load class names
f = open('gesture.names', 'r')
classNames = f.read().split('\n')
f.close()
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
    stato = statoDita(landmarks, mano, verso)
    quante = 0
    for i, s in enumerate(stato):
        if i !=0 and s:
            quante+=1
    return quante

#conta il numero di dita
def statoDita(landmarks, mano, verso):
    fingersUp = [False, False, False, False, False]

    #pollice (dipende dalla mano e dal verso della mano)
    #v1: funziona solo se la mano NON è in pos.orizzontale
    if verso.lower() == "palm":
        if (mano.lower() == "left" and landmarks[4][0] > landmarks[3][0]) or (mano.lower() == "right" and landmarks[4][0] < landmarks[3][0]):
            fingersUp[0] = True
    elif verso.lower() == "back":
        if (mano.lower() == "left" and landmarks[4][0] < landmarks[3][0]) or (mano.lower() == "right" and landmarks[4][0] > landmarks[3][0]):
            fingersUp[0] = True

    #indice
    if landmarks[8][1] < landmarks[6][1]:
        fingersUp[1] = True
    #medio
    if landmarks[12][1] < landmarks[10][1]:
        fingersUp[2] = True
    #anulare
    if landmarks[16][1] < landmarks[14][1]: 
        fingersUp[3] = True
    #mignolo
    if landmarks[20][1] < landmarks[18][1]:   
        fingersUp[4] = True

    return fingersUp

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
drawingPoints = []
colors = [(255, 255, 255), (0, 255, 255), (0, 0, 255), (255, 0, 0), (0, 255, 0)]
colorNames = ("white", "yellow", "red", "blue", "green")
thickness = 3
color = 0
segments = {}
segment = 1
drawModeswitches = 0
enableThicknessSwitch = 0
enableDrawModeSwitch = True
enableColorSwitch = True
colorSwitches = 0

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


            # Predict gesture
            prediction = model.predict([landmarks], verbose = False)
            # print(prediction)
            classID = np.argmax(prediction)
            className = classNames[classID]



            #increase or decrease thickness
            if className == "thumbs up":
                if enableThicknessSwitch == 0:
                    thickness += 1
                drawingPoints = []
                if drawMode:
                    segments["segment"+str(segment)] = (drawingPoints, colors[color], thickness)
                    segment += 1
                enableDrawModeSwitch = False
                enableColorSwitch = False
            elif className == "thumbs down":
                if thickness > 0 and enableThicknessSwitch == 0:
                    thickness -= 1
                    drawingPoints = []
                    if drawMode:
                        segments["segment"+str(segment)] = (drawingPoints, colors[color], thickness)
                        segment += 1
                enableDrawModeSwitch = False
                enableColorSwitch = False
            else:
                enableDrawModeSwitch = True
                enableColorSwitch = True

            #switch on/off drawing mode
            if numeroDita(landmarks, etichettaMano, verso(landmarks, etichettaMano)) == 1 and statoDita(landmarks, etichettaMano, verso(landmarks, etichettaMano))[1]:
                if enableDrawModeSwitch:
                    drawMode = True
                    drawModeswitches+=1
                    if drawMode and drawModeswitches == 1:
                        segments["segment"+str(segment)] = (drawingPoints, colors[color], thickness)
                        segment += 1
            else:
                if enableDrawModeSwitch:
                    drawMode = False
                    drawModeswitches = 0
                    drawingPoints = []

#            if numeroDita(landmarks, etichettaMano, verso(landmarks, etichettaMano))[1]:
#                if enableDrawModeSwitch:
#                    drawMode = True
#                    drawModeswitches+=1
#                    if drawMode and drawModeswitches == 1:
#                        segments["segment"+str(segment)] = (drawingPoints, colors[color], thickness)
#                        segment += 1
#            else:
#                if enableDrawModeSwitch:
#                    drawMode = False
#                    drawModeswitches = 0
#                    drawingPoints = []

            #switch between colors
            if statoDita(landmarks, etichettaMano, verso(landmarks, etichettaMano))[0]:
                if enableColorSwitch:
                    colorSwitches += 1
                    if colorSwitches == 1:
                        if color == -5:
                            color = 0
                        color -= 1
                        drawingPoints = []
                        if drawMode:
                            segments["segment"+str(segment)] = (drawingPoints, colors[color], thickness)
                            segment += 1
                    enableDrawModeSwitch = False
            else:
                colorSwitches = 0
                enableDrawModeSwitch = False

            #erase screen
            if statoDita(landmarks, etichettaMano, verso(landmarks, etichettaMano))[1] and statoDita(landmarks, etichettaMano, verso(landmarks, etichettaMano))[2]:
                if abs(landmarks[8][0] - landmarks[12][0]) < 30 and abs(landmarks[8][1] - landmarks[12][1]) < 30:
                    enableDrawModeSwitch = False
                    enableColorSwitch = False
                    segments = {}
                    segment = 1
            else:
                enableDrawModeSwitch = True
                enableColorSwitch = True

            # Drawing landmarks on frames
            if drawMode:
                cv2.rectangle(frame, (0,0), (1280, 720), (0,0,0), 999)
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

    else:
        drawMode = False
        drawingPoints = []
        drawModeswitches = 0

    #Disegno partendo dall'indice    
    for i in range(len(segments)):
        points = [np.array(segments["segment"+str(i + 1)][0])]
        lineColor = segments["segment"+str(i + 1)][1]
        lineThickness = segments["segment"+str(i + 1)][2]
        cv2.polylines(frame, points, False, lineColor, lineThickness)

        
    cv2.putText(frame, "Draw Mode: "+("enabled" if drawMode else "disabled"), [10, 25], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if drawMode else (0, 0, 255), 2)
    cv2.putText(frame, "Color: "+colorNames[color], [10, 60], cv2.FONT_HERSHEY_SIMPLEX, 1, colors[color], 2)
    cv2.putText(frame, "Thickness: "+str(thickness), [10, 95], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # # show the prediction on the frame
    # cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
    #                1, (0,0,255), 2, cv2.LINE_AA)

    # Show the final output
    cv2.imshow("Output", frame) 

    if cv2.waitKey(1) == ord('q'):
        break
    
    enableThicknessSwitch += 1
    if enableThicknessSwitch == 7:
        enableThicknessSwitch = 0
    
    
    

# release the webcam and destroy all active windows
cap.release()

cv2.destroyAllWindows()