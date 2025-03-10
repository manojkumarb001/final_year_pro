# import cv2
# import mediapipe as mp
# import numpy as np
# from OpenGL.GL import *
# from OpenGL.GLUT import *
# from OpenGL.GLU import *

# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands()

# def draw_hand(landmarks):
#     glBegin(GL_POINTS)
#     for lm in landmarks:
#         glVertex3f(lm.x, lm.y, lm.z)  # Render hand points in 3D
#     glEnd()

# def render():
#     glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
#     glLoadIdentity()

#     # Capture hand
#     ret, frame = cap.read()
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = hands.process(rgb_frame)

#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             draw_hand(hand_landmarks.landmark)

#     glutSwapBuffers()

# cap = cv2.VideoCapture(0)
# glutInit()
# glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
# glutCreateWindow("3D Hand")
# glutDisplayFunc(render)
# glutMainLoop()


import json
import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np

# Load stored hand landmark data
with open("hand_landmarks.json", "r") as f:
    hand_data = json.load(f)

if not hand_data:
    print("No hand tracking data found!")
    exit()

# Initialize MediaPipe hand connections
mp_hands = mp.solutions.hands
connections = mp_hands.HAND_CONNECTIONS

# Video settings
video_filename = "hand_animation1.mp4"
frame_width = 500
frame_height = 500
fps = 20

# Initialize OpenCV VideoWriter
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4
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
