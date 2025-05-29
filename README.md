# AI-based Traffic Light Control System

An intelligent traffic management system that uses computer vision to dynamically control traffic light timings based on real-time vehicle density detection.

## Overview

This project implements an adaptive traffic light control system that uses computer vision to:
- Detect and count vehicles in real-time from camera feeds
- Calculate vehicle density for each lane
- Dynamically adjust traffic light durations based on traffic density
- Provide a visual simulation of the traffic flow

## Tech Stack

### Core Components
- **Object Detection**: YOLOv5s (optimized for edge devices)
- **Model Optimization**:
  - ONNX Runtime (CPU/GPU compatible)
  - TensorRT (GPU-optimized)
- **Hardware Requirements**:
  - NVIDIA Jetson Nano (recommended)
  - IP Cameras
  - GPU-enabled system (for TensorRT implementation)

### Performance Comparison (on Jetson Nano)

| Detection Algorithm | Platform  | FPS   |
|---------------------|-----------|-------|
| YOLOv5s            | PyTorch   | 3.125 |
| YOLOv5s            | ONNX      | 4     |
| YOLOv5s            | TensorRT  | 13    |

## Project Structure
```
project/
│   README.md
│   requirements.txt    
│   app.py
│
├── common/
│   └── utils.py
│
├── data/
│   ├── video.mp4
│   ├── video1.mp4
│   ├── video2.mp4
│   ├── video3.mp4
│   ├── video4.mp4
│   └── coco.names
│
├── implementation_with_yolov5s_onnx_model/
│   └── main.py
│
├── implementation_with_yolov5s_tensorrt_model/
│   ├── processor.py
│   └── main.py
│
├── models/
│   ├── yolov5s.onnx
│   └── yolov5s.trt
│
└── templates/
    └── index.html
```

## System Architecture

The system follows this workflow:
1. Camera feeds capture traffic from multiple lanes
2. YOLOv5s model detects vehicles in real-time
3. Vehicle counting and density calculation per lane
4. Dynamic traffic light timing adjustment
5. Visual simulation of traffic flow

## Installation

1. Clone the repository:
```bash
git clone https://github.com/gm0202/TrafficController.git
cd TrafficController
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download required models:
- YOLOv5s ONNX model
- YOLOv5s TensorRT model (for GPU implementation)

## Usage

### ONNX Implementation (CPU/GPU)
This implementation works on both CPU and GPU environments:
```bash
cd implementation_with_yolov5s_onnx_model
python main.py --sources video1.mp4,video2.mp4,video3.mp4,video4.mp4
```

### TensorRT Implementation (GPU Only)
For optimal performance on NVIDIA GPUs:
```bash
cd implementation_with_yolov5s_tensorrt_model
python main.py --sources video1.mp4,video2.mp4,video3.mp4,video4.mp4
```

### Web Interface
To run the web interface:
```bash
python app.py
```

## Model Optimization

The project uses two optimized versions of YOLOv5s:
1. **ONNX Model**: 
   - Compatible with both CPU and GPU
   - Moderate performance improvement
   - Easier deployment

2. **TensorRT Model**:
   - GPU-optimized for maximum performance
   - Significant speed improvement (4x faster than PyTorch)
   - Requires NVIDIA GPU

## References

1. YOLOv5 Model Export to ONNX:
   - [YOLOv5 GitHub Repository](https://github.com/ultralytics/yolov5)

2. ONNX to TensorRT Conversion:
   - [YOLOv5 TensorRT Guide](https://github.com/SeanAvery/yolov5-tensorrt)

## Contributing

This is an ongoing research project. Contributions are welcome! Please feel free to submit issues and pull requests.

## License

This project is open source and available under the MIT License.
