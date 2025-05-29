from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import time
from common.utils import final_output, Lanes, Lane, display_result, schedule

app = Flask(__name__)

# Global variables for status tracking
fps = 0
processing_time = 0
current_lane = 1
wait_time = 0
traffic_light_status = 'red'
processed_lanes = None
frame_count = 0  # For frame skipping
last_frame_time = 0  # For FPS calculation

# Traffic light timing constants
MIN_GREEN_TIME = 5  # Minimum green light duration in seconds
MAX_GREEN_TIME = 30  # Maximum green light duration in seconds
YELLOW_TIME = 3  # Yellow light duration in seconds

# Frame processing settings
TARGET_FPS = 30
FRAME_SKIP_THRESHOLD = 1.0 / 15  # Skip frames if processing takes longer than 1/15 second
INPUT_WIDTH = 640  # Reduced input size for faster processing
INPUT_HEIGHT = 480

# Load the ONNX model
print("Loading ONNX model...")
net = cv2.dnn.readNet("implementation_with_yolov5s_onnx_model/yolov5s.onnx")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)  # Use OpenCV backend
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)       # Use CPU target
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
print("Model loaded successfully")

def calculate_green_time(vehicles):
    """Calculate green light duration based on vehicle count and priorities"""
    if not vehicles:
        return MIN_GREEN_TIME
    
    # Priority weights for different vehicle types
    priority_weights = {
        'car': 1,
        'truck': 2,
        'bus': 2,
        'motorcycle': 0.5
    }
    
    # Calculate weighted vehicle count
    weighted_count = sum(priority_weights.get(vehicle.type, 1) for vehicle in vehicles)
    
    # Calculate green time based on weighted count
    # More vehicles = shorter time, but with minimum and maximum limits
    green_time = MAX_GREEN_TIME - (weighted_count * 2)  # 2 seconds reduction per weighted vehicle
    green_time = max(MIN_GREEN_TIME, min(MAX_GREEN_TIME, green_time))
    
    return green_time

def get_vehicle_counts(lanes):
    counts = {
        'cars': 0,
        'trucks': 0,
        'buses': 0,
        'motorcycles': 0
    }
    
    if lanes and lanes.getLanes():
        for lane in lanes.getLanes():
            if lane.vehicles:
                for vehicle in lane.vehicles:
                    if vehicle.type == 'car':
                        counts['cars'] += 1
                    elif vehicle.type == 'truck':
                        counts['trucks'] += 1
                    elif vehicle.type == 'bus':
                        counts['buses'] += 1
                    elif vehicle.type == 'motorcycle':
                        counts['motorcycles'] += 1
    
    return counts

def generate_frames():
    global fps, processing_time, current_lane, wait_time, traffic_light_status, processed_lanes, frame_count, last_frame_time
    
    # Initialize single camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, INPUT_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, INPUT_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer size
    
    # Initialize single lane
    lanes = Lanes([])
    lanes.enque(Lane(0, None, 1))
    
    wait_time = 0
    last_update = time.time()
    current_lane = 1
    traffic_light_status = 'red'
    yellow_light_start = 0
    
    while True:
        # Read frame from camera
        success, frame = cap.read()
        if not success:
            continue
            
        # Skip frames if processing is too slow
        current_time = time.time()
        if current_time - last_frame_time < FRAME_SKIP_THRESHOLD:
            continue
            
        # Resize frame for faster processing
        frame = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT))
        
        # Update lane with new frame
        lanes.getLanes()[0].frame = frame
        
        # Process frame with the model
        start = time.time()
        processed_lanes = final_output(net, ln, lanes)
        end = time.time()
        
        # Update global performance metrics
        processing_time = (end - start) * 1000  # Convert to milliseconds
        fps = 1 / (end - start)
        last_frame_time = current_time
        
        # Calculate traffic light timing
        if current_time - last_update >= 1:  # Update every second
            if traffic_light_status == 'yellow':
                # Check if yellow light duration is over
                if current_time - yellow_light_start >= YELLOW_TIME:
                    traffic_light_status = 'red'
                    wait_time = 0
            else:
                wait_time = max(0, wait_time - 1)
                if wait_time <= 0:
                    if traffic_light_status == 'red':
                        # Calculate new green time based on current vehicles
                        current_vehicles = processed_lanes.getLanes()[0].vehicles if processed_lanes and processed_lanes.getLanes() else []
                        wait_time = calculate_green_time(current_vehicles)
                        traffic_light_status = 'green'
                        print(f"Green light for {wait_time:.1f} seconds based on {len(current_vehicles)} vehicles")
                    else:  # green
                        traffic_light_status = 'yellow'
                        yellow_light_start = current_time
                        print("Switching to yellow light")
            last_update = current_time
        
        # Display results
        display_frame = display_result(wait_time, processed_lanes)
        
        # Add FPS and processing time
        cv2.putText(display_frame, f"FPS: {fps:.2f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Processing Time: {processing_time:.1f}ms", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Convert to JPEG with lower quality for faster transmission
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
        ret, buffer = cv2.imencode('.jpg', display_frame, encode_param)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def get_status():
    global fps, processing_time, current_lane, wait_time, traffic_light_status, processed_lanes
    return jsonify({
        'fps': fps,
        'processing_time': processing_time,
        'current_lane': current_lane,
        'time_remaining': wait_time,
        'traffic_light': traffic_light_status,
        'vehicle_counts': get_vehicle_counts(processed_lanes)
    })

if __name__ == '__main__':
    app.run(debug=True) 