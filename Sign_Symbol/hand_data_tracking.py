import cv2
import mediapipe as mp
import json
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Start Webcam
cap = cv2.VideoCapture(0)
hand_data = []  # Stores multiple hands per frame

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Flip for a natural view
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    frame_landmarks = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in hand_landmarks.landmark]
            frame_landmarks.append(landmarks)  # Append each hand separately

    if frame_landmarks:
        hand_data.append(frame_landmarks)  # Store multiple hands in a single frame

    # Display Camera Feed
    cv2.imshow("Hand Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Save Data to JSON
with open("hand_landmarks.json", "w") as f:
    json.dump(hand_data, f, indent=4)

cap.release()
cv2.destroyAllWindows()

print("✅ Hand tracking data saved to 'hand_landmarks.json'")
print("----------------------------------------------------------")





# Load JSON file
with open("hand_landmarks.json", "r") as f:
    hand_data = json.load(f)

# Apply transformation to 3D space
scaled_hand_data = []

for frame in hand_data:
    transformed_frame = []
    for hand in frame:
        transformed_hand = []
        for lm in hand:
            transformed_hand.append({
                "x": (lm["x"] - 0.5) * 2,  # Convert to range (-1,1)
                "y": (lm["y"] - 0.5) * 2,  # Convert to range (-1,1)
                "z": lm["z"] * 100         # Scale depth for visibility
            })
        transformed_frame.append(transformed_hand)
    scaled_hand_data.append(transformed_frame)

# Save the modified data
with open("transformed_hand_landmarks.json", "w") as f:
    json.dump(scaled_hand_data, f, indent=4)

print("✅ Transformed hand data saved to 'transformed_hand_landmarks.json'")
