import json
import numpy as np
import open3d as o3d
import cv2
import time

# Load Hand Landmarks Data
with open("transformed_hand_landmarks.json", "r") as f:
    hand_data = json.load(f)

# Check if hand_data is valid
if not hand_data:
    print("⚠ No valid hand data found in JSON file. Exiting...")
    exit()

print(f"✅ Loaded {len(hand_data)} frames of hand tracking data.")

# Initialize Open3D Visualizer
vis = o3d.visualization.Visualizer()
vis.create_window(width=800, height=600)

# Configure Open3D Rendering
opt = vis.get_render_option()
opt.point_size = 15  # Increase point size for better visibility
opt.background_color = np.array([0, 0, 0])  # Black background for contrast

# Create a point cloud object
pcd = o3d.geometry.PointCloud()
vis.add_geometry(pcd)

# Get camera view control and set a fixed view angle
view_ctl = vis.get_view_control()
view_ctl.set_zoom(0.7)  # Adjust zoom level for better view

# Video Settings
video_filename = "hand_animation.mp4"
frame_width, frame_height = 800, 600
fps = 20

# Initialize OpenCV Video Writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = cv2.VideoWriter(video_filename, fourcc, fps, (frame_width, frame_height))

# Ensure visualization stabilizes before recording
for _ in range(5):
    vis.poll_events()
    vis.update_renderer()
    time.sleep(0.1)

# Process Each Frame for Video
for frame_idx, frame_data in enumerate(hand_data):
    if not frame_data:
        continue

    # Extract 3D hand points
    points = np.array([[lm["x"], lm["y"], lm["z"] * 200] for hand in frame_data for lm in hand])  # Scale Z-axis

    if len(points) > 0:
        # Generate random colors for visualization
        colors = np.random.rand(len(points), 3)

        # Update point cloud
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        vis.update_geometry(pcd)

        # Render multiple times to prevent flickering
        for _ in range(3):
            vis.poll_events()
            vis.update_renderer()

        # Capture the frame after rendering stabilizes
        time.sleep(0.05)
        img = (np.asarray(vis.capture_screen_float_buffer(True)) * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (frame_width, frame_height))  # Ensure size consistency

        # Save frame to video
        video_writer.write(img)

        print(f"📸 Processed Frame {frame_idx + 1}/{len(hand_data)}")

# Release video writer
video_writer.release()
print(f"✅ 3D animation saved as '{video_filename}'")

# Process the LAST frame to save a final 3D image
final_frame = hand_data[-1]  # Use the last recorded frame

if final_frame:
    points = np.array([[lm["x"], lm["y"], lm["z"] * 200] for hand in final_frame for lm in hand])

    if len(points) > 0:  # Ensure we have valid points
        # Generate random colors
        colors = np.random.rand(len(points), 3)

        # Update point cloud with points and colors
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        vis.update_geometry(pcd)
        for _ in range(10):  # Ensure rendering updates
            vis.poll_events()
            vis.update_renderer()

        # Delay before capturing
        time.sleep(0.5)

        # Capture and save the final rendered 3D image
        final_image = (np.asarray(vis.capture_screen_float_buffer(True)) * 255).astype(np.uint8)
        final_image = cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite("final_3d_hand.png", final_image)
        print("✅ Final 3D Image saved as 'final_3d_hand.png'")

    else:
        print("⚠ No points found in the final frame.")
else:
    print("⚠ Final frame is empty.")

# Release Open3D resources
vis.destroy_window()
