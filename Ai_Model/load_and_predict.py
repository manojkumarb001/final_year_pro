# import os
# import cv2
# import numpy as np
# from tensorflow.keras.models import load_model

# # Load the trained model
# model = load_model("sign_language_model.h5")

# # Load label names
# labels = sorted(os.listdir("D:/Indian"))

# def predict_sign(image_path):
#     image = cv2.imread(image_path)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image = cv2.resize(image, (64, 64)) / 255.0
#     image = np.expand_dims(image, axis=0)

#     prediction = model.predict(image)
#     predicted_class = np.argmax(prediction)

#     print(f"🖐 Predicted Sign: {labels[predicted_class]}")

# # Example usage
# predict_sign(r"D:/Indian/2/14.jpg")
# predict_sign(r"D:/615.jpg")

# predict_sign(r"D:/9.jpg")
# predict_sign(r"D:/img1.jpeg")
print("---------------------------------")

import cv2
import numpy as np
import os
import mediapipe as mp
import time
from tensorflow.keras.models import load_model
from textwrap import fill

# Load the trained model
model = load_model("sign_language_model.h5")

# Load label names
labels = sorted(os.listdir("D:/Indian"))

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False,
                        max_num_hands=2,  # Track up to two hands
                        min_detection_confidence=0.7,
                        min_tracking_confidence=0.7)

# Variables to store recognized text
recognized_sentence = ""
last_prediction_time = time.time()  # Track last prediction time
tracking_start_time = None  # Start time for tracking hands
current_sign = ""  # Store the currently predicted sign

# Function to predict sign from extracted hand region
def predict_sign_from_frame(hand_img):
    try:
        hand_img_rgb = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)
        hand_img_resized = cv2.resize(hand_img_rgb, (64, 64)) / 255.0
        hand_img_expanded = np.expand_dims(hand_img_resized, axis=0)

        prediction = model.predict(hand_img_expanded)
        predicted_class = np.argmax(prediction)

        return labels[predicted_class]
    except Exception as e:
        return "Unknown"

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("❌ Error: Unable to capture video")
        break

    frame = cv2.flip(frame, 1)  # Mirror effect
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    hand_predictions = []  # Store detected signs from hands
    hands_detected = False  # Track if hands are detected

    if results.multi_hand_landmarks:
        hands_detected = True  # At least one hand is detected

        for hand_landmarks in results.multi_hand_landmarks:
            # Get hand bounding box
            h, w, _ = frame.shape
            x_min, y_min = w, h
            x_max, y_max = 0, 0

            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min, y_min = min(x, x_min), min(y, y_min)
                x_max, y_max = max(x, x_max), max(y, y_max)

            # Expand bounding box slightly
            x_min, x_max = max(0, x_min - 20), min(w, x_max + 20)
            y_min, y_max = max(0, y_min - 20), min(h, y_max + 20)

            # Extract hand region
            hand_img = frame[y_min:y_max, x_min:x_max]

            if hand_img.size != 0:
                predicted_sign = predict_sign_from_frame(hand_img)
                hand_predictions.append(predicted_sign)

                # Draw bounding box and landmarks
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Display predicted sign
                cv2.putText(frame, predicted_sign, (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    current_time = time.time()

    # If hands are detected, start tracking time
    if hands_detected:
        if tracking_start_time is None:
            tracking_start_time = current_time
        elif current_time - tracking_start_time >= 2:  # 2 seconds delay
            if hand_predictions:
                if len(hand_predictions) == 1:  # Single-hand sign
                    current_sign = hand_predictions[0]
                elif len(hand_predictions) == 2:  # Two-hand sign (treated as a phrase)
                    current_sign = "_".join(hand_predictions)

                recognized_sentence += current_sign + " "
                tracking_start_time = None  # Reset tracking time after adding to sentence

    else:
        tracking_start_time = None  # Reset if hands are lost

    # Wrap the sentence text
    wrapped_sentence = fill(recognized_sentence, width=40)  # Wrap text after 40 characters

    # Display wrapped sentence
    y_offset = 50
    for line in wrapped_sentence.split("\n"):
        cv2.putText(frame, line, (50, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        y_offset += 30  # Adjust line spacing

    cv2.imshow("Sign Language Recognition", frame)

    # Press 's' to save the sentence to a file
    if cv2.waitKey(1) & 0xFF == ord("s"):
        with open("recognized_text.txt", "w") as f:
            f.write(recognized_sentence.strip())
        print("✅ Sentence saved to 'recognized_text.txt'")

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
