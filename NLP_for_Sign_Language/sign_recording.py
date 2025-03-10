import cv2
import json
import os
import time
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=2)
    # Directory to store sign recordings and hand tracking data
output_dir = "recorded_signs"
os.makedirs(output_dir, exist_ok=True)

    # JSON file to store mappings
json_file = "sign_mappings.json"
sign_data = {}

    # Load existing data if available
if os.path.exists(json_file):
    with open(json_file, "r") as f:
        sign_data = json.load(f)
def gesture_recognize():
        
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=2)

    # Directory to store sign recordings and hand tracking data
    output_dir = "recorded_signs"
    os.makedirs(output_dir, exist_ok=True)

    # JSON file to store mappings
    json_file = "sign_mappings.json"
    sign_data = {}

    # Load existing data if available
    if os.path.exists(json_file):
        with open(json_file, "r") as f:
            sign_data = json.load(f)

    # Start capturing video
    cap = cv2.VideoCapture(0)

    while True:
        word = input("\nEnter the word for the sign (or 'q' to quit): ").strip().lower()
        if word == 'q':
            break

        print(f"📢 Recording sign for '{word}' in 3 seconds...")
        time.sleep(3)  # Countdown before recording

        video_filename = os.path.join(output_dir, f"{word}.mp4")
        hand_data_filename = os.path.join(output_dir, f"{word}_landmarks.json")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(video_filename, fourcc, 10, (640, 480))

        start_time = time.time()
        hand_tracking_data = []

        while time.time() - start_time < 3:  # Record for 3 seconds
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame for a natural view
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            frame_landmarks = []

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    landmarks = [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in hand_landmarks.landmark]
                    frame_landmarks.append(landmarks)

            if frame_landmarks:
                hand_tracking_data.append(frame_landmarks)

            out.write(frame)
            cv2.putText(frame, f"Recording '{word}'", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Recording Sign", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        out.release()

        # Save hand tracking data
        with open(hand_data_filename, "w") as f:
            json.dump(hand_tracking_data, f, indent=4)

        sign_data[word] = {"video": video_filename, "landmarks": hand_data_filename}
        print(f"✅ Saved '{word}' as {video_filename} with hand tracking data.")

    # Save mappings to JSON
    with open(json_file, "w") as f:
        json.dump(sign_data, f, indent=4)

    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Sign recordings complete. Data saved to sign_mappings.json.")

# Function to generate and play sign animation
def play_sign_animation(word):
    if word not in sign_data:
        print(f"❌ No sign found for '{word}'")
        return

    hand_data_path = sign_data[word]["landmarks"]

    # Load stored hand landmark data
    with open(hand_data_path, "r") as f:
        hand_data = json.load(f)

    if not hand_data:
        print("❌ No hand tracking data available!")
        return

    # Initialize MediaPipe hand connections
    connections = mp_hands.HAND_CONNECTIONS

    # Video settings
    frame_width = 500
    frame_height = 500
    fps = 20

    # Initialize OpenCV VideoWriter
    video_filename = os.path.join(output_dir, f"{word}_animation.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(video_filename, fourcc, fps, (frame_width, frame_height))

    # Function to draw hand landmarks
    def draw_hand(landmarks, image):
        for start, end in connections:
            pt1 = (int(landmarks[start]["x"] * frame_width), int(landmarks[start]["y"] * frame_height))
            pt2 = (int(landmarks[end]["x"] * frame_width), int(landmarks[end]["y"] * frame_height))
            cv2.line(image, pt1, pt2, (0, 255, 0), 2)  # Green lines

        for lm in landmarks:
            x, y = int(lm["x"] * frame_width), int(lm["y"] * frame_height)
            cv2.circle(image, (x, y), 5, (255, 0, 0), -1)  # Blue points

    # Process each frame
    for frame_data in hand_data:
        if not frame_data:
            continue

        frame = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 255  # White background

        for hand in frame_data:  # Iterate over all detected hands
            draw_hand(hand, frame)

        video_writer.write(frame)  # Save frame to video

    video_writer.release()  # Save the video file
    print(f"✅ Animation saved as {video_filename}")

    # Play the animated video
    cap = cv2.VideoCapture(video_filename)

    if not cap.isOpened():
        print(f"❌ Failed to open {video_filename}")
        return

    print("🎬 Playing animation. Press 'q' to exit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Animated Sign", frame)

        # If 'q' is pressed, break the loop
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Continuous input loop to play signs
# while True:
#     word_to_play = input("\nEnter a word to generate its hand animation (or 'q' to quit): ").strip().lower()
#     if word_to_play == 'q':
#         print("✅ Exiting the program.")
#         break

#     play_sign_animation(word_to_play)
