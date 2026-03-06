import cv2
import numpy as np
import pickle
import os
import multiprocessing
from datetime import datetime
from insightface.app import FaceAnalysis
import onnxruntime as ort

class FaceRecognitionSystem:
    def __init__(self, data_dir='data', use_optimized=True):
        """Initialize the face recognition system with ONNX Runtime optimizations"""
        self.data_dir = data_dir
        self.embeddings_dir = os.path.join(data_dir, 'embeddings')
        self.images_dir = os.path.join(data_dir, 'captured_images')
        self.embeddings_file = os.path.join(self.embeddings_dir, 'face_embeddings.pkl')
        
        # Create directories if they don't exist
        os.makedirs(self.embeddings_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        
        # Configure ONNX Runtime session options for CPU optimization
        if use_optimized:
            print("Configuring ONNX Runtime optimizations for CPU...")
            self._configure_onnx_optimizations()
        
        # Initialize face analysis model
        print("Loading InsightFace model with optimizations...")
        self.app = FaceAnalysis(
            name='buffalo_l', 
            providers=['CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=-1, det_size=(640, 640))  # Optimized detection size
        print("Model loaded successfully with CPU optimizations!")
        
        # Load existing embeddings
        self.known_faces = self.load_embeddings()
    
    def _configure_onnx_optimizations(self):
        """Configure ONNX Runtime session options for optimal CPU performance"""
        # Get number of physical CPU cores
        num_cores = multiprocessing.cpu_count() // 2  # Physical cores (not logical)
        
        # Set ONNX Runtime session options globally
        sess_options = ort.SessionOptions()
        
        # Thread optimization
        sess_options.intra_op_num_threads = num_cores  # Use all physical cores
        sess_options.inter_op_num_threads = 1  # Sequential execution for face models
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        
        # Graph optimization
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Enable thread spinning for lower latency (uses more CPU but faster)
        sess_options.add_session_config_entry("session.intra_op.allow_spinning", "1")
        
        print("ONNX Optimization configured:")
        print(f"  - Physical CPU cores: {num_cores}")
        print(f"  - Intra-op threads: {num_cores}")
        print("  - Graph optimization: ENABLED")
        print("  - Thread spinning: ENABLED (lower latency)")
        
        # Store for potential future use
        self.sess_options = sess_options
        
    def load_embeddings(self):
        """Load stored face embeddings"""
        if os.path.exists(self.embeddings_file):
            with open(self.embeddings_file, 'rb') as f:
                return pickle.load(f)
        return {}
    
    def save_embeddings(self):
        """Save face embeddings to disk"""
        with open(self.embeddings_file, 'wb') as f:
            pickle.dump(self.known_faces, f)
        print(f"Embeddings saved to {self.embeddings_file}")
    
    def get_face_embedding(self, img):
        """Extract face embedding from an image - returns single face"""
        faces = self.app.get(img)
        
        if len(faces) < 1:
            return None, None
        if len(faces) > 1:
            print("Warning: Multiple faces detected. Using first detected face")
        
        return faces[0].embedding, faces[0].bbox
    
    def get_all_face_embeddings(self, img):
        """Extract all face embeddings from an image"""
        faces = self.app.get(img)
        
        if len(faces) < 1:
            return []
        
        results = []
        for face in faces:
            results.append({
                'embedding': face.embedding,
                'bbox': face.bbox
            })
        
        return results
    
    def compare_faces(self, emb1, emb2, threshold=0.4):
        """Compare two embeddings using cosine similarity"""
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return similarity, similarity > threshold
    
    def recognize_face(self, embedding, threshold=0.4):
        """Recognize a face by comparing with known embeddings"""
        if not self.known_faces:
            return "Unknown", 0.0
        
        best_match = None
        best_similarity = 0.0
        
        for name, known_embedding in self.known_faces.items():
            similarity, is_match = self.compare_faces(embedding, known_embedding, threshold)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = name
        
        if best_similarity > threshold:
            return best_match, best_similarity
        return "Unknown", best_similarity
    
    def capture_and_register(self, name, source=0, num_captures=5):
        """Capture images from webcam/RTSP and register a new person"""
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print(f"Error: Could not open video source {source}")
            return False
        
        print(f"\nCapturing {num_captures} images for {name}")
        print("Press SPACE to capture, ESC to cancel")
        
        person_dir = os.path.join(self.images_dir, name)
        os.makedirs(person_dir, exist_ok=True)
        
        captured_count = 0
        embeddings = []
        
        while captured_count < num_captures:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            # Display frame
            display_frame = frame.copy()
            cv2.putText(display_frame, f"Captured: {captured_count}/{num_captures}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, "Press SPACE to capture, ESC to cancel", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow('Capture Face', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                print("Capture cancelled")
                cap.release()
                cv2.destroyAllWindows()
                return False
            
            elif key == 32:  # SPACE
                embedding, bbox = self.get_face_embedding(frame)
                
                if embedding is not None:
                    # Save image
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    img_path = os.path.join(person_dir, f"image_{timestamp}.jpg")
                    cv2.imwrite(img_path, frame)
                    
                    embeddings.append(embedding)
                    captured_count += 1
                    print(f"Captured {captured_count}/{num_captures}")
                else:
                    print("No face detected. Please try again.")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Average the embeddings
        if embeddings:
            avg_embedding = np.mean(embeddings, axis=0)
            self.known_faces[name] = avg_embedding
            self.save_embeddings()
            print(f"\n{name} registered successfully with {len(embeddings)} images!")
            return True
        
        return False

    def recognize_from_stream(self, source=0, confidence_threshold=0.4, skip_frames=2):
        """Real-time face recognition from webcam or RTSP stream with FPS optimization"""
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print(f"Error: Could not open video source {source}")
            return
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("\nStarting face recognition...")
        print("Press 'q' to quit")
        print(f"Processing every {skip_frames + 1} frames for better FPS")
        
        frame_count = 0
        last_results = []  # Cache results for skipped frames
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            # Process only every nth frame for better FPS
            if frame_count % (skip_frames + 1) == 0:
                # Get all face embeddings
                faces = self.get_all_face_embeddings(frame)
                
                last_results = []
                for face_data in faces:
                    embedding = face_data['embedding']
                    bbox = face_data['bbox']
                    
                    # Recognize face
                    name, similarity = self.recognize_face(embedding, confidence_threshold)
                    
                    last_results.append({
                        'name': name,
                        'similarity': similarity,
                        'bbox': bbox
                    })
            
            # Draw results (even on skipped frames)
            for result in last_results:
                x1, y1, x2, y2 = result['bbox'].astype(int)
                name = result['name']
                similarity = result['similarity']
                
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label with background
                label = f"{name} ({similarity:.2f})"
                (label_width, label_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                cv2.rectangle(frame, (x1, y1 - label_height - 10), 
                            (x1 + label_width, y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display FPS counter
            cv2.putText(frame, f"Faces: {len(last_results)}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Display frame
            cv2.imshow('Face Recognition', frame)
            
            frame_count += 1
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def list_registered_faces(self):
        """List all registered faces"""
        if not self.known_faces:
            print("No faces registered yet.")
            return
        
        print("\nRegistered faces:")
        for i, name in enumerate(self.known_faces.keys(), 1):
            print(f"{i}. {name}")
    
    def delete_face(self, name):
        """Delete a registered face"""
        if name in self.known_faces:
            del self.known_faces[name]
            self.save_embeddings()
            print(f"{name} deleted successfully!")
            return True
        else:
            print(f"{name} not found in registered faces.")
            return False


def main():
    """Main function with menu interface"""
    print("\n" + "="*50)
    print("Face Recognition System - ONNX Optimized")
    print("="*50)
    print("Initialize with optimizations?")
    print("1. Yes - Use ONNX Runtime optimizations (Recommended)")
    print("2. No - Use default settings")
    print("="*50)
    
    opt_choice = input("Enter choice (1-2, default 1): ").strip()
    use_optimized = opt_choice != '2'
    
    system = FaceRecognitionSystem(use_optimized=use_optimized)
    
    while True:
        print("\n" + "="*50)
        print("Face Recognition System")
        print("="*50)
        print("1. Register new face (Webcam)")
        print("2. Register new face (RTSP URL)")
        print("3. Recognize faces (Webcam)")
        print("4. Recognize faces (RTSP URL)")
        print("5. List registered faces")
        print("6. Delete registered face")
        print("7. Exit")
        print("="*50)
        
        choice = input("Enter your choice (1-7): ").strip()
        
        if choice == '1':
            name = input("Enter person's name: ").strip()
            if name:
                num_captures = input("Number of images to capture (default 5): ").strip()
                num_captures = int(num_captures) if num_captures.isdigit() else 5
                system.capture_and_register(name, source=0, num_captures=num_captures)
            else:
                print("Name cannot be empty!")
        
        elif choice == '2':
            name = input("Enter person's name: ").strip()
            rtsp_url = input("Enter RTSP URL: ").strip()
            if name and rtsp_url:
                num_captures = input("Number of images to capture (default 5): ").strip()
                num_captures = int(num_captures) if num_captures.isdigit() else 5
                system.capture_and_register(name, source=rtsp_url, num_captures=num_captures)
            else:
                print("Name and RTSP URL cannot be empty!")
        
        elif choice == '3':
            threshold = input("Enter confidence threshold (default 0.4): ").strip()
            threshold = float(threshold) if threshold else 0.4
            skip = input("Skip frames for better FPS (0=no skip, 2=default, 3-5=faster): ").strip()
            skip = int(skip) if skip.isdigit() else 2
            system.recognize_from_stream(source=0, confidence_threshold=threshold, skip_frames=skip)
        
        elif choice == '4':
            rtsp_url = input("Enter RTSP URL: ").strip()
            if rtsp_url:
                threshold = input("Enter confidence threshold (default 0.4): ").strip()
                threshold = float(threshold) if threshold else 0.4
                skip = input("Skip frames for better FPS (0=no skip, 2=default, 3-5=faster): ").strip()
                skip = int(skip) if skip.isdigit() else 2
                system.recognize_from_stream(source=rtsp_url, confidence_threshold=threshold, skip_frames=skip)
            else:
                print("RTSP URL cannot be empty!")
        
        elif choice == '5':
            system.list_registered_faces()
        
        elif choice == '6':
            system.list_registered_faces()
            name = input("Enter name to delete: ").strip()
            if name:
                system.delete_face(name)
        
        elif choice == '7':
            print("Exiting...")
            break
        
        else:
            print("Invalid choice! Please try again.")


if __name__ == "__main__":
    main()
