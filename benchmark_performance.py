"""
Benchmark script to compare performance with and without ONNX optimizations
"""
import cv2
import time
import numpy as np
from face_recognition_system import FaceRecognitionSystem

def benchmark_inference(system, num_frames=100):
    """Benchmark face detection and recognition performance"""
    print(f"\nBenchmarking with {num_frames} frames...")
    
    # Create a test image (or use webcam)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam. Using synthetic test image.")
        # Create synthetic test image
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cap = None
    else:
        ret, test_frame = cap.read()
        if not ret:
            print("Error: Could not read from webcam")
            return None
    
    times = []
    successful_detections = 0
    
    print("Running benchmark...")
    for i in range(num_frames):
        if cap is not None:
            ret, frame = cap.read()
            if not ret:
                frame = test_frame
        else:
            frame = test_frame
        
        start_time = time.time()
        
        # Perform face detection and embedding extraction
        faces = system.get_all_face_embeddings(frame)
        
        # If faces detected, perform recognition
        if faces:
            successful_detections += 1
            for face_data in faces:
                embedding = face_data['embedding']
                name, similarity = system.recognize_face(embedding, threshold=0.4)
        
        end_time = time.time()
        times.append(end_time - start_time)
        
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{num_frames} frames...")
    
    if cap is not None:
        cap.release()
    
    # Calculate statistics
    avg_time = np.mean(times) * 1000  # Convert to ms
    std_time = np.std(times) * 1000
    min_time = np.min(times) * 1000
    max_time = np.max(times) * 1000
    fps = 1.0 / np.mean(times)
    
    return {
        'avg_time_ms': avg_time,
        'std_time_ms': std_time,
        'min_time_ms': min_time,
        'max_time_ms': max_time,
        'fps': fps,
        'successful_detections': successful_detections,
        'total_frames': num_frames
    }

def main():
    print("="*60)
    print("InsightFace ONNX Runtime Optimization Benchmark")
    print("="*60)
    
    num_frames = 50  # Reduced for faster testing
    
    # Benchmark WITHOUT optimizations
    print("\n[1/2] Testing WITHOUT ONNX optimizations...")
    print("-" * 60)
    system_default = FaceRecognitionSystem(use_optimized=False)
    results_default = benchmark_inference(system_default, num_frames)
    
    # Small delay between tests
    time.sleep(2)
    
    # Benchmark WITH optimizations
    print("\n[2/2] Testing WITH ONNX optimizations...")
    print("-" * 60)
    system_optimized = FaceRecognitionSystem(use_optimized=True)
    results_optimized = benchmark_inference(system_optimized, num_frames)
    
    # Display results
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    
    if results_default and results_optimized:
        print("\nWithout ONNX Optimizations:")
        print(f"  Average inference time: {results_default['avg_time_ms']:.2f} ms")
        print(f"  Std deviation: {results_default['std_time_ms']:.2f} ms")
        print(f"  Min/Max time: {results_default['min_time_ms']:.2f} / {results_default['max_time_ms']:.2f} ms")
        print(f"  Average FPS: {results_default['fps']:.2f}")
        print(f"  Successful detections: {results_default['successful_detections']}/{results_default['total_frames']}")
        
        print("\nWith ONNX Optimizations:")
        print(f"  Average inference time: {results_optimized['avg_time_ms']:.2f} ms")
        print(f"  Std deviation: {results_optimized['std_time_ms']:.2f} ms")
        print(f"  Min/Max time: {results_optimized['min_time_ms']:.2f} / {results_optimized['max_time_ms']:.2f} ms")
        print(f"  Average FPS: {results_optimized['fps']:.2f}")
        print(f"  Successful detections: {results_optimized['successful_detections']}/{results_optimized['total_frames']}")
        
        # Calculate improvement
        speedup = results_default['avg_time_ms'] / results_optimized['avg_time_ms']
        fps_improvement = ((results_optimized['fps'] - results_default['fps']) / results_default['fps']) * 100
        
        print("\n" + "="*60)
        print("PERFORMANCE IMPROVEMENT")
        print("="*60)
        print(f"  Speedup: {speedup:.2f}x faster")
        print(f"  FPS improvement: {fps_improvement:.1f}%")
        print(f"  Time saved per frame: {results_default['avg_time_ms'] - results_optimized['avg_time_ms']:.2f} ms")
        print("="*60)
        
        if speedup > 1.2:
            print("\n✓ Significant performance improvement achieved!")
        elif speedup > 1.0:
            print("\n✓ Moderate performance improvement achieved!")
        else:
            print("\n⚠ No significant improvement. This may vary by hardware.")
    else:
        print("\nBenchmark failed. Please check your setup.")

if __name__ == "__main__":
    main()
