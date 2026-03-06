"""
Quick test script to verify ONNX optimizations are working
"""
import cv2
import time
from face_recognition_system import FaceRecognitionSystem

def quick_test():
    print("="*60)
    print("Quick ONNX Optimization Test")
    print("="*60)
    
    # Test with optimizations
    print("\nInitializing with ONNX optimizations...")
    system = FaceRecognitionSystem(use_optimized=True)
    
    # Try to open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("\nError: Could not open webcam")
        print("Please ensure your webcam is connected and not in use by another application")
        return
    
    print("\nTesting face detection performance...")
    print("Look at the camera. Press 'q' to quit.\n")
    
    frame_times = []
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Measure inference time
        inference_start = time.time()
        faces = system.get_all_face_embeddings(frame)
        inference_time = (time.time() - inference_start) * 1000  # Convert to ms
        
        frame_times.append(inference_time)
        frame_count += 1
        
        # Display results
        for face_data in faces:
            bbox = face_data['bbox']
            x1, y1, x2, y2 = bbox.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Calculate and display FPS
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        avg_inference = sum(frame_times[-30:]) / len(frame_times[-30:]) if frame_times else 0
        
        # Display stats
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Inference: {avg_inference:.1f}ms", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Faces: {len(faces)}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "ONNX Optimized", (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        cv2.imshow('ONNX Optimization Test', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Print summary
    if frame_times:
        print("\n" + "="*60)
        print("Test Results")
        print("="*60)
        print(f"Total frames processed: {frame_count}")
        print(f"Average FPS: {fps:.2f}")
        print(f"Average inference time: {sum(frame_times)/len(frame_times):.2f} ms")
        print(f"Min inference time: {min(frame_times):.2f} ms")
        print(f"Max inference time: {max(frame_times):.2f} ms")
        print("="*60)
        
        if fps > 15:
            print("\n✓ Excellent performance! Optimizations are working well.")
        elif fps > 10:
            print("\n✓ Good performance! Optimizations are active.")
        else:
            print("\n⚠ Lower than expected performance. Check CPU usage and system load.")

if __name__ == "__main__":
    quick_test()
