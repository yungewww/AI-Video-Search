'''
STEP 2.1
Input:
- Videos
Output:
- Capture screenshots
- Preprocess Screenshots, naming: [vidId, frameNum, timestamp]
'''

import cv2
import numpy as np
import os

# Load video
video_paths = [
    'videos/How Green Roofs Can Help Cities  NPR.mp4',
    'videos/What Does High-Quality Preschool Look Like  NPR Ed.mp4',
    'videos/Why It’s Usually Hotter In A City  Lets Talk  NPR.mp4'
]

output_dir = './screenshots'
os.makedirs(output_dir, exist_ok=True)

all_decoded_frames = []
all_frame_info = []

frame_sample_rate = 100
target_height = 360
target_width = 640

for i, path in enumerate(video_paths):
    cap = cv2.VideoCapture(path)
    decoded_frames = []
    frame_info = []  # To store frame information

    # Process video
    frame_count = 0  # To keep track of the frame number
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # No more frames or error

        # For each frame
        if frame_count % frame_sample_rate == 0:
            # Preprocess
            resized_frame = cv2.resize(frame, (target_width, target_height)) # Resize frame
            frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB) # Convert color from BGR to RGB
            normalized_frame = frame_rgb / 255.0 # Normalize pixel values to the range 0-1

            # Append to list
            decoded_frames.append(normalized_frame)
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # Get timestamp in seconds
            frame_info.append((i+1, frame_count, timestamp))

            # Save the frame
            output_filename = os.path.join(output_dir, f"{i + 1}_{frame_count}_{timestamp}.jpg")
            cv2.imwrite(output_filename, resized_frame)

            print(f"VidID: {i+1}, Frame Count: {frame_count} , Time Stamp: {timestamp}")

        frame_count += 1

    cap.release()
    all_decoded_frames.extend(decoded_frames)
    all_frame_info.extend(frame_info)

# Convert list of frames to numpy array and save
np.save('frames.npy', np.array(all_decoded_frames))

# Also save frame information (frame number and timestamp)
frame_info_np = np.array(all_frame_info, dtype=[('vidID','i4'), ('frameNum', 'i4'), ('timestamp', 'f4')])
np.save('frame_info.npy', frame_info_np)
