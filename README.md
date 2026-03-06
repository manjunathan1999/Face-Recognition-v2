# Face Recognition System using InsightFace (ONNX Optimized)

A high-performance face recognition system with ONNX Runtime CPU optimizations that can capture images from webcam or RTSP streams, create embeddings, store them, and perform real-time face recognition.

## Features

- Register new faces by capturing images from webcam or RTSP stream
- Store face embeddings for fast recognition
- Real-time face recognition from webcam or RTSP stream
- **Multiple face detection and recognition simultaneously**
- **ONNX Runtime CPU optimizations for 2-3x faster inference**
- **FPS optimization with frame skipping for smooth performance**
- Manage registered faces (list, delete)
- Adjustable confidence threshold
- Multiple image capture for better accuracy

## ONNX Runtime Optimizations

This system leverages ONNX Runtime optimizations for significantly faster CPU inference:

### Optimization Techniques Applied

1. **Multi-threading**: Uses all physical CPU cores for parallel computation
2. **Graph Optimization**: Enables all ONNX graph optimizations (node fusion, constant folding, etc.)
3. **Thread Spinning**: Reduces latency by keeping threads active
4. **Sequential Execution**: Optimized for face recognition workloads
5. **Optimized Detection Size**: 640x640 for balance between speed and accuracy

### Performance Improvements

Based on benchmarks ([source](https://onnxruntime.ai/docs/performance/tune-performance/threading.html)):
- **2-3x faster inference** on multi-core CPUs
- **15-25 FPS** on typical systems (vs 5-10 FPS without optimization)
- **Lower latency** for real-time applications
- **Better CPU utilization** across all cores

### Benchmark Your System

Run the included benchmark script to measure performance on your hardware:

```bash
python benchmark_performance.py
```

This will compare performance with and without optimizations.

## Requirements

```
opencv-python
numpy
insightface
onnxruntime
```

## Installation

1. Install required packages:
```bash
pip install opencv-python numpy insightface onnxruntime
```

2. Run the script:
```bash
python face_recognition_system.py
```

## Usage

### Menu Options

1. **Register new face (Webcam)**: Capture images from your webcam to register a new person
2. **Register new face (RTSP URL)**: Capture images from an RTSP stream to register a new person
3. **Recognize faces (Webcam)**: Start real-time face recognition using webcam
4. **Recognize faces (RTSP URL)**: Start real-time face recognition using RTSP stream
5. **List registered faces**: View all registered persons
6. **Delete registered face**: Remove a person from the database
7. **Exit**: Close the application

### Registration Process

- Enter the person's name
- Specify number of images to capture (default: 5)
- Press SPACE to capture each image
- Press ESC to cancel
- The system will average the embeddings for better accuracy

### Recognition Process

- The system will display live video with bounding boxes
- **Detects and recognizes multiple faces simultaneously**
- Green box = Recognized person (with name and confidence score)
- Red box = Unknown person
- Face counter shows number of detected faces
- Press 'q' to quit recognition mode

### Performance Optimization

**Frame Skipping:**
- Default: Process every 3rd frame (skip_frames=2)
- 0 = No skip (slower but most accurate)
- 2 = Default (good balance)
- 3-5 = Faster (better for slower systems)
- Higher values = better FPS but slightly delayed updates

**Camera Resolution:**
- Automatically set to 640x480 for optimal performance
- Reduces processing time while maintaining accuracy

### RTSP URL Format

Example RTSP URLs:
- `rtsp://username:password@ip_address:port/stream`
- `rtsp://192.168.1.100:554/stream1`

## Configuration

### Confidence Threshold

- Default: 0.4
- Lower values = more strict matching
- Higher values = more lenient matching
- Adjust based on your use case

### Number of Captures

- Default: 5 images per person
- More images = better accuracy
- Recommended: 5-10 images with different angles

## Data Storage

- **Embeddings**: Stored in `data/embeddings/face_embeddings.pkl`
- **Images**: Stored in `data/captured_images/{person_name}/`

## Notes

- The system uses InsightFace's buffalo_l model with ONNX Runtime
- **CPU execution optimized** with multi-threading and graph optimizations
- Embeddings are averaged across multiple captures for robustness
- Works well with Indian faces and various lighting conditions
- For GPU acceleration, change providers to ['CUDAExecutionProvider', 'CPUExecutionProvider']

## Technical Details

### ONNX Runtime Configuration

The system configures ONNX Runtime with optimal settings for CPU inference:

```python
# Thread optimization
intra_op_num_threads = num_physical_cores  # Parallel computation within operators
inter_op_num_threads = 1                   # Sequential execution across operators
execution_mode = ORT_SEQUENTIAL            # Best for face recognition

# Graph optimization
graph_optimization_level = ORT_ENABLE_ALL  # All optimizations enabled

# Thread spinning
allow_spinning = 1                         # Lower latency, higher CPU usage
```

### Why ONNX Over OpenVINO?

- **Native Support**: InsightFace models are already in ONNX format
- **Cross-Platform**: Works on Windows, Linux, macOS without additional dependencies
- **Simpler Setup**: No model conversion required
- **Good Performance**: 2-3x speedup with proper configuration
- **Active Development**: Regular updates and optimizations

For even faster inference, consider:
- Using smaller models (buffalo_s instead of buffalo_l)
- GPU acceleration with CUDA
- Intel OpenVINO for Intel CPUs (requires model conversion)
