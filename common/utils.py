import numpy as np
import cv2
import time
import pathlib


"""
a blueprint for a bounded box with its corresponding name,confidence score and 
"""
print(pathlib.Path.cwd())

class BoundedBox:
    
    def __init__(self, xmin, ymin, xmax, ymax, ids, confidence):
        with open(str(pathlib.Path.cwd()) + "/datas/coco.names", 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')  # stores a list of classes
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax       
        self.ymax = ymax
        self.name = self.classes[ids]
        self.confidence = confidence   


"""
a blueprint that has lanes as lists and give queue like functionality 
to reorder lanes based on their turn for green and red light state
"""

class Lanes:
    def __init__(self,lanes):
        self.lanes=lanes
    
    def getLanes(self):
        
        return self.lanes
    
    def lanesTurn(self):
        
       return self.lanes.pop(0)

    def enque(self,lane):
 
       return self.lanes.append(lane)
    def lastLane(self):
       return self.lanes[len(self.lanes)-1]
"""
a blueprint that has lanes as lists and give queue like functionality 
to reorder lanes based on their turn for green and red light state
"""
class Lane:
    def __init__(self,count,frame,lane_number):
        self.count = count
        self.frame = frame
        self.lane_number = lane_number
    
"""
given lanes object return a duration based on comparison of each lane vehicle count
"""
def schedule(lanes):
    # Calculate priority for each lane
    lane_priorities = []
    for lane in lanes.getLanes():
        priority = calculate_lane_priority(lane.vehicles)
        lane_priorities.append((lane, priority))
    
    # Sort lanes by priority
    lane_priorities.sort(key=lambda x: x[1], reverse=True)
    
    # Calculate base time (minimum time for any lane)
    base_time = 10  # minimum 10 seconds
    
    # Calculate time for each lane based on priority
    scheduled_times = {}
    for i, (lane, priority) in enumerate(lane_priorities):
        # Higher priority lanes get more time
        time = base_time + (priority * 2)  # 2 seconds per priority level
        scheduled_times[lane.lane_number] = time
    
    return scheduled_times
       
"""
given duration and lanes, returns a grid image containing frames of each lane with
their corresponding waiting duration
"""   

def display_result(wait_time,lanes):
    green = (0,255,0)
    red  = (0,0,255)
    yellow= (0,255,255)
    lane_imgs = []
    for i ,lane in enumerate(lanes.getLanes()):
        #resized so that all images have the same dimension inorder to be concatenable
        lane.frame = cv2.resize(lane.frame,(1280, 720)) 
        if(wait_time<=0 and (i==(len(lanes.getLanes())-1) or i==0)):
           color=yellow
           text="yellow:2 sec"
        elif(wait_time>=0 and i==(len(lanes.getLanes())-1)):
            color = green 
            text="green:"+str(wait_time)+" sec"
        else:
            color=red
            text="red:"+str(wait_time)+ " sec"
        lane.frame = cv2.putText(lane.frame,text,(60,105),cv2.FONT_HERSHEY_SIMPLEX,4,color,6)
        lane.frame = cv2.putText(lane.frame,"vehicle count:"+str(lane.count),(60,195),cv2.FONT_HERSHEY_SIMPLEX,3,color,5)
        lane_imgs.append(lane.frame)
    # Handle different number of lanes
    if len(lane_imgs) == 1:
        return lane_imgs[0]
    elif len(lane_imgs) == 2:
        return np.concatenate((lane_imgs[0], lane_imgs[1]), axis=1)
    elif len(lane_imgs) == 4:
        hori_image = np.concatenate((lane_imgs[0], lane_imgs[1]), axis=1)
        hori2_image = np.concatenate((lane_imgs[2], lane_imgs[3]), axis=1)
        all_lanes_image = np.concatenate((hori_image, hori2_image), axis=0)
        return all_lanes_image
    else:
        # Fallback: just return the first image
        return lane_imgs[0]



# given detecteed boxes, return number of vehicles on each box
def vehicle_count(Boxes):
    vehicles = []
    for box in Boxes:
        if box.name in ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'person']:
            vehicles.append(Vehicle(box.name, box.confidence))
    return vehicles

# given the grid dimension, returns a 2d grid
def _make_grid(nx=20, ny=20):
        xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
        return np.stack((xv, yv), 2).reshape((1, 1, ny, nx, 2)).astype(np.float32)

def drawPred( frame, classId, conf, left, top, right, bottom):
        
       
        # Draw a bounding box.
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), thickness=6)

        return frame

def modify(outs, confThreshold=0.5, nmsThreshold=0.5):
    # outs may be a tuple or list, get the first array
    if isinstance(outs, (tuple, list)):
        outs = outs[0]
    if hasattr(outs, 'shape') and len(outs.shape) == 3:
        outs = outs[0]  # shape: (N, 85)
    boxes = []
    confidences = []
    classIds = []
    for detection in outs:
        scores = detection[5:]
        classId = np.argmax(scores)
        confidence = scores[classId] * detection[4]  # class conf * obj conf
        if confidence > confThreshold:
            center_x = int(detection[0])
            center_y = int(detection[1])
            width = int(detection[2])
            height = int(detection[3])
            left = int(center_x - width / 2)
            top = int(center_y - height / 2)
            boxes.append([left, top, width, height])
            confidences.append(float(confidence))
            classIds.append(classId)
    # NMS
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    detections = []
    for i in indices:
        i = i[0] if isinstance(i, (list, tuple, np.ndarray)) else i
        detections.append((boxes[i], classIds[i], confidences[i]))
    return detections

def postprocess(frame, detections):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    # Load class names
    with open(str(pathlib.Path.cwd())+'/datas/coco.names', 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')
    correct_boxes = []
    for box, classId, confidence in detections:
        left, top, width, height = box
        right = left + width
        bottom = top + height
        # Clamp coordinates
        left = max(0, left)
        top = max(0, top)
        right = min(frameWidth - 1, right)
        bottom = min(frameHeight - 1, bottom)
        # Draw
        frame = drawPred(frame, classId, confidence, left, top, right, bottom)
        correct_boxes.append(BoundedBox(left, top, right, bottom, classId, confidence))
    return correct_boxes, frame


"""
given each lanes image, it inferences using trt engine on the image, return lanes object
containg processed image and waiting duration for each image

"""
def final_output_tensorrt(processor,lanes):
     
    for lane in lanes.getLanes():
            lane.frame=cv2.resize(lane.frame,(1280,720))      #resize into a standard image  dimension
            start = time.time()
            output = processor.detect(lane.frame)
            end = time.time() 
            print("fps:"+str(end-start))   
            dets = modify(output)
            boxes,frame = postprocess(lane.frame,dets)
            count = vehicle_count(boxes)
            lane.count= count
            lane.frame=frame
            
        
        
    return lanes

"""
given each lanes image, it inferences onnx model on the image, return lanes object
containg processed image and waiting duration for each image

"""

def final_output(net, output_layer, lanes):
    for lane in lanes.getLanes():
        # Resize frame for model input
        lane.frame = cv2.resize(lane.frame, (640, 640))
        
        # Create blob and run inference
        blob = cv2.dnn.blobFromImage(lane.frame, 1/255.0, (640, 640),
                                   swapRB=True, crop=False)
        net.setInput(blob)
        
        # Run inference
        start = time.time()
        layerOutputs = net.forward(output_layer)
        end = time.time()
        print(f"FPS: {1/(end-start):.2f}")
        
        # Process detections
        dets = modify(layerOutputs)
        boxes, frame = postprocess(lane.frame, dets)
        
        # Count vehicles and store them
        lane.vehicles = vehicle_count(boxes)
        lane.count = len(lane.vehicles)
        lane.frame = frame
        
    return lanes

class Vehicle:
    def __init__(self, vehicle_type, confidence):
        self.type = vehicle_type
        self.confidence = confidence
        self.priority = self._get_priority()

    def _get_priority(self):
        priorities = {
            'ambulance': 5,
            'fire truck': 5,
            'police car': 5,
            'truck': 4,
            'bus': 4,
            'car': 3,
            'motorcycle': 2,
            'bicycle': 2,
            'auto-rickshaw': 1,
            'person': 1
        }
        return priorities.get(self.type, 0)

def calculate_lane_priority(vehicles):
    if not vehicles:
        return 0
    
    # Base priority is the highest priority vehicle in the lane
    base_priority = max(vehicle.priority for vehicle in vehicles)
    
    # Count vehicles by type
    vehicle_counts = {}
    for vehicle in vehicles:
        vehicle_counts[vehicle.type] = vehicle_counts.get(vehicle.type, 0) + 1
    
    # Calculate weighted priority based on vehicle counts
    weighted_priority = base_priority
    for vehicle_type, count in vehicle_counts.items():
        vehicle = Vehicle(vehicle_type, 1.0)  # Create temporary vehicle for priority
        weighted_priority += (vehicle.priority * count * 0.1)  # 0.1 is a scaling factor
    
    return weighted_priority
