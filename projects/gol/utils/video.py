import cv2
import numpy as np
from tqdm import tqdm

def video(frames, output_file = 'output.avi', fps = 10.0, res_increase_factor = 10):
	# Get the shape of the frames
	frames = np.array(frames)
	height, width = frames[0].shape
	height *= res_increase_factor
	width *= res_increase_factor

	# Create a VideoWriter object
	fourcc = cv2.VideoWriter_fourcc(*'FFV1')  # Use 'FFV1' for lossless video
	video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

	# Loop through the saved frames
	print("Writing video...")
	for frame in tqdm(frames):
		# Read the frame
		frame = cv2.cvtColor(np.uint8(frame), cv2.COLOR_GRAY2BGR) * 255
		
		# Increase resolution
		frame = cv2.resize(frame, None, fx=res_increase_factor, fy=res_increase_factor, interpolation=cv2.INTER_NEAREST)
		
		# Write the frame to the video file
		video.write(frame)

	# Release the VideoWriter
	video.release()
	print("Video saved as", output_file)