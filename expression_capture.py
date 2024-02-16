import cv2
import os
import mediapipe as mp

# Path where all the images will be stored
IMG_PATH = "./data/test"

# Defining some constants
WAIT_FRAMES = 5
CAP_FRAMES = 20

frame_counter: int = 0
wait_counter: int = 0
iteration: int = -1

# Get the mediapipe model
Hand_landmark_model = mp.solutions.hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Define the video capture device | If not working try a different number instead of 0
cap = cv2.VideoCapture(0)


while True:

    # Capture a frame
    ret, frame = cap.read()
    # Convert from BGR to RGB
    rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run the landmark model on the RGB image
    results = Hand_landmark_model.process(rgb_img)

    if results.multi_handedness:

        if frame_counter == 0 and wait_counter <= WAIT_FRAMES:

            if wait_counter == WAIT_FRAMES:
                frame_counter = 1
                wait_counter = 0

                iteration += 1

                temp_path = os.path.join(IMG_PATH, str(iteration))

                os.mkdir(temp_path)

            else:
                text = "WAITING!"

                cv2.putText(frame, text, (50, 50), cv2.FONT_ITALIC, 1, (0, 200, 0), 2, cv2.LINE_AA)

                wait_counter += 1

        if CAP_FRAMES >= frame_counter > 0:

            text = "CAPTURING GESTURE....."

            cv2.imwrite(os.path.join(temp_path, str(frame_counter-1) + ".jpg"), frame)

            cv2.putText(frame, text, (50, 50), cv2.FONT_ITALIC, 1, (0, 200, 0), 2, cv2.LINE_AA)

            frame_counter += 1
        if results.multi_handedness and frame_counter > CAP_FRAMES:
            text = "GESTURE CAPTURED!"

            cv2.putText(frame, text, (50, 50), cv2.FONT_ITALIC, 1, (0, 200, 0), 2, cv2.LINE_AA)
    else:
        frame_counter = 0

    cv2.imshow("frame", frame)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
