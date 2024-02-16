import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model

WAIT_FRAMES = 5
CAP_FRAMES = 20

frame_counter: int = 0
wait_counter: int = 0

mp_hands = mp.solutions.hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
mp_drawing = mp.solutions.drawing_utils

classifier = load_model('BSL_Expressions.keras')

cap = cv2.VideoCapture(0)

expressions = []
sample = []
time_stamp = []
norm_x = []
norm_y = []

while True:

    success, frame = cap.read()

    rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = mp_hands.process(rgb_img)

    if results.multi_handedness and frame_counter < CAP_FRAMES:

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                norm_x.append(x)
                norm_y.append(y)

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x = x - min(norm_x)
                y = y - min(norm_y)

                time_stamp.append(x)
                time_stamp.append(y)

        sample.append(time_stamp)
        frame_counter += 1
        time_stamp = []

        if frame_counter == CAP_FRAMES:

            expressions.append(sample)

            exp = np.asarray(expressions)

            pred = classifier.predict(exp)

            print(pred)

            expressions = []

            norm_x = []
            norm_y = []
            sample = []

    else:
        frame_counter = 0


    cv2.imshow('frame', frame)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()



