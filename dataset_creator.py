import cv2
import mediapipe as mp
import pickle
import os

# Import the mediapipe model for hand landmarks detection. Static images
mp_hands = mp.solutions.hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Path where all the data is stored
DATA_PATH = "./data"

expressions = []
labels = []

for expression_dir in os.listdir(DATA_PATH):
    for sample_dir in os.listdir(os.path.join(DATA_PATH, expression_dir)):
        sample = []
        for time_step_dir in os.listdir(os.path.join(DATA_PATH, expression_dir, sample_dir)):
            time_step = []
            # Loop from 0 to 19, so that all the features are in order.
            for i in range(20):

                filename = os.path.join(DATA_PATH, expression_dir, sample_dir, time_step_dir, str(i), ".jpg")

                # Load the image and then convert it from BGR to RGB
                img = cv2.imread(os.path.join(DATA_PATH, expression_dir, sample_dir, time_step_dir, filename))
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Process the image to obtain the landmarks position (x,y)
                results = mp_hands.process(rgb_img)

                # Loop through all the landmarks
                for hand_landmarks in results.multi_hand_landmarks:
                    for j in range(len(hand_landmarks.landmark)):

                        x = hand_landmarks.landmark[j].x
                        y = hand_landmarks.landmark[j].y

                        time_step.append(x)
                        time_step.append(y)

        sample.append(time_step)
    expressions.append(sample)
    labels.append(expression_dir)

print(expressions)





