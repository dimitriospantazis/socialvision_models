import os
import fnmatch
import subprocess
import functools
import re
import subprocess
import shutil
import numpy as np
import ffmpeg
import io
import numpy as np
from PIL import Image
import cv2  # OpenCV for video processing


# List of target videos inside a directory
def find_videos(directory, target_names=None):
    video_extensions = ['*.mp4'] # Add more extensions if needed
    # Convert the target names to a set if provided, for efficient lookups
    target_set = set(target_names) if target_names is not None else None
    
    matched_videos = []
    for root, _, files in os.walk(directory):
        for ext in video_extensions:
            for filename in fnmatch.filter(files, ext):
                # If no target list is provided, return all discovered videos
                # Otherwise, return only those that are in the target_set
                if target_set is None or filename in target_set:
                    matched_videos.append(os.path.join(root, filename))
    
    return matched_videos



def extract_frames(video_path, num_frames):
    """
    Extracts equispaced frames from video

    Parameters:
    - video_path: The path of the video to search for (e.g., 'my_video.mp4').
    - num_frames: The number of frames

    Returns:
    - PIL images.
    """  
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get the total number of frames and the frame rate (frames per second)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if num_frames > total_frames:
        raise ValueError("Number of frames requested exceeds total frames in the video.")

    # Calculate the interval in frames to extract equispaced frames
    interval = max(1, total_frames // num_frames)

    frames = []  # List to store extracted PIL images

    # Iterate through the video, extracting frames at the computed interval
    for i in range(num_frames):
        frame_number = i * interval  # Compute the frame number to extract
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)  # Set position
        ret, frame = cap.read()  # Read the frame

        if ret:
            # Convert frame from BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert the NumPy array to a PIL Image
            pil_image = Image.fromarray(frame_rgb)
            frames.append(pil_image)
        else:
            print(f"Could not extract frame at {frame_number}. Skipping...")

    cap.release()  # Release the video capture object
    return frames




