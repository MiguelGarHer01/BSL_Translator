import cv2
import mediapipe as mp
import numpy as np
import os
import keras

# Import the mediapipe model for hand landmarks detection. Static images
mp_hands = mp.solutions.hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Path where all the data is stored
DATA_PATH = "./test"

expressions = []
dir = []

print("Loading data...")

print("Processing data...")

for expression_dir in os.listdir(DATA_PATH):
    for sample_dir in os.listdir(os.path.join(DATA_PATH, expression_dir)):
        sample = []
        # Loop from 0 to 19, so that all the features are in order.
        for i in range(20):
            time_step = []

            # Added two lists to normalize the position of the hand in the frame
            norm_x = []
            norm_y = []
            norm_z = []
            count = 0

            filename = os.path.join(DATA_PATH, expression_dir, sample_dir, str(i) + ".jpg")

            # Load the image and then convert it from BGR to RGB
            img = cv2.imread(filename)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Process the image to obtain the landmarks position (x,y)
            results = mp_hands.process(rgb_img)

            # Loop through all the landmarks
            for hand_landmarks in results.multi_hand_landmarks:
                if len(results.multi_handedness) == 1:

                    for j in range(len(hand_landmarks.landmark)):

                        x = hand_landmarks.landmark[j].x
                        y = hand_landmarks.landmark[j].y
                        z = hand_landmarks.landmark[j].z

                        # Save the coordinates for a future normalization
                        norm_x.append(x)
                        norm_y.append(y)
                        norm_z.append(z)

                    for j in range(len(hand_landmarks.landmark)):

                        x = hand_landmarks.landmark[j].x
                        y = hand_landmarks.landmark[j].y
                        z = hand_landmarks.landmark[j].z

                        # NORMALIZATION
                        x = x - min(norm_x)
                        y = y - min(norm_y)
                        z = z - min(norm_z)

                        # Save the coordinates as features in our time_step
                        time_step.append(x)
                        time_step.append(y)
                        time_step.append(z)

                    # Save the time steps into our sample expression
                    sample.append(time_step)

                else:

                    if count < 21:

                        for j in range(len(hand_landmarks.landmark)):
                            x = hand_landmarks.landmark[j].x
                            y = hand_landmarks.landmark[j].y
                            z = hand_landmarks.landmark[j].z

                            # Save the coordinates for a future normalization
                            norm_x.append(x)
                            norm_y.append(y)
                            norm_z.append(z)

                        for j in range(len(hand_landmarks.landmark)):

                            x = hand_landmarks.landmark[j].x
                            y = hand_landmarks.landmark[j].y
                            z = hand_landmarks.landmark[j].z

                            # NORMALIZATION
                            x = x - min(norm_x)
                            y = y - min(norm_y)
                            z = z - min(norm_z)

                            # Save the coordinates as features in our time_step
                            time_step.append(x)
                            time_step.append(y)
                            time_step.append(z)

                            count += 1

                        # Save the time steps into our sample expression
                        sample.append(time_step)

        # Save the sample of the expression into the expression list
        expressions.append(sample)
        dir.append(sample_dir)

classifier = keras.models.load_model('BSL_Expressions.keras')

expressions = np.asarray(expressions)

predictions = classifier.predict(expressions)
print(predictions)
print(dir)

