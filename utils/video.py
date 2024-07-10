import cv2
import sys
import os
from data.constant import PHASES
from tqdm import tqdm

class VideoGenerator:
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.video_length = 1000
        
    def generate_video(self, output_path):
        # Sort the image paths
        sorted_paths = []
        for phase in PHASES:
            sorted_paths.extend(sorted([path for path in self.image_paths if phase in path]))
        
        downsample_factor = len(sorted_paths) // self.video_length
        sorted_paths = sorted_paths[::downsample_factor]
        
        # Get the dimensions of the first image
        first_image = cv2.imread(sorted_paths[0])
        height, width, _ = first_image.shape

        # Create a video writer object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, 30, (width, height))

        # Iterate over the sorted image paths and write each frame to the video
        for image_path in tqdm(sorted_paths):
            frame = cv2.imread(image_path)
            video_writer.write(frame)

        # Release the video writer and destroy any remaining windows
        video_writer.release()
        cv2.destroyAllWindows()