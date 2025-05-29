import sys
import argparse
import pathlib
sys.path.insert(0, str(pathlib.Path.cwd() / "common"))

import cv2
import utils as util
import time
import numpy as np


def main(sources):
	print("Starting main function...")
	
	# Initialize video capture (use camera if no video files provided)
	if sources[0].lower() == 'camera':
		print("Using camera input...")
		vs = cv2.VideoCapture(0)  # Use default camera
	else:
		print(f"Opening video source: {sources[0]}")
		vs = cv2.VideoCapture(sources[0])

	# Check if video opened successfully
	if not vs.isOpened():
		print("Error: Could not open video source")
		return

	print("Loading ONNX model...")
	#creates a network given yolov5s model
	net = cv2.dnn.readNet("models/yolov5s.onnx")
	# Since we don't have CUDA, use CPU
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
	ln = net.getUnconnectedOutLayersNames() # returns the name of output layer
	print("Model loaded successfully")

	# Initialize single lane
	lanes = util.Lanes([util.Lane("", "", 1)])
	
	# Initialize traffic light states
	current_lane = 1
	wait_time = 0
	scheduled_times = {}

	print("Starting main loop...")
	while True:
		# Read frame
		success, frame = vs.read()
		if not success:
			print("Error: Could not read frame from source")
			break

		# Assign frame to lane
		lanes.getLanes()[0].frame = frame

		# Process frame and detect vehicles
		start = time.time()
		lanes = util.final_output(net, ln, lanes)
		end = time.time()
		print(f"Total processing time: {end-start:.2f} seconds")

		# Update traffic light timing based on priorities
		if wait_time <= 0:
			scheduled_times = util.schedule(lanes)
			current_lane = max(scheduled_times.items(), key=lambda x: x[1])[0]
			wait_time = scheduled_times[current_lane]
			print(f"Lane {current_lane} has highest priority. Time: {wait_time} seconds")

		# Display results
		images = util.display_result(wait_time, lanes)
		final_image = cv2.resize(images, (1020, 720))
		
		# Add priority information to display
		for lane in lanes.getLanes():
			priority = util.calculate_lane_priority(lane.vehicles)
			cv2.putText(final_image, f"Priority: {priority:.1f}", 
					   (60, 285), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

		cv2.imshow("Traffic Control System", final_image)
		
		# Handle key press
		key = cv2.waitKey(1)
		if key == ord('q'):
			print("Quitting...")
			break

		wait_time -= 1

	# Cleanup
	vs.release()
	cv2.destroyAllWindows()

if __name__=="__main__":
	parser = argparse.ArgumentParser(description="AI-based Traffic Control System")
	parser.add_argument("--sources", type=str, 
					   default="camera",
					   help="Video source (camera or video file)")
	args = parser.parse_args()

	sources = args.sources.split(",")
	print("Using video source:", sources[0])
	main(sources)		        



