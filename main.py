import cv2
import face_recognition
import numpy as np
import os
import pickle
import time
import threading
from queue import Queue, Empty
from collections import deque
import logging

class EliteFaceRecognition:
    def __init__(self, tolerance=0.6, model='hog', debug=False):
        """
        Initialize the Face Recognition System
        
        Args:
            tolerance (float): Threshold for face matching (0.4-0.6 recommended)
            model (str): 'hog' for speed, 'cnn' for accuracy
            debug (bool): Enable debug logging
        """
        self.tolerance = tolerance
        self.model = model
        self.debug = debug
        
        # Setup logging
        logging.basicConfig(level=logging.DEBUG if debug else logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Face data storage
        self.known_encodings = []
        self.known_names = []
        
        # Performance optimization settings
        self.PROCESS_EVERY_N_FRAMES = 3
        self.RESIZE_FACTOR = 4
        self.MAX_FACES_PER_FRAME = 5
        self.ENCODING_JITTERS = 3  # Reduced for speed
        self.UPSAMPLE_TIMES = 1 if model == 'hog' else 0
        
        # Frame processing variables
        self.frame_count = 0
        self.last_face_locations = []
        self.last_face_names = []
        self.last_face_confidences = []
        self.confidence_history = {}  # For temporal smoothing
        self.MIN_CONFIDENCE = 75      # Minimum confidence threshold for known faces
        
        # Performance tracking
        self.fps_tracker = deque(maxlen=30)
        self.processing_times = deque(maxlen=10)
        
        # Thread-safe processing
        self.processing_lock = threading.Lock()
        self.processing_thread = None
        self.processing_queue = Queue(maxsize=2)
        self.result_queue = Queue(maxsize=2)
        self.stop_processing = threading.Event()
        
        self.logger.info(f" Face Recognition System initialized")
        self.logger.info(f"Settings: tolerance={tolerance}, model={model}")
    
    def validate_folder_structure(self, faces_folder_path):
        """Validate the folder structure and return person folders"""
        if not os.path.exists(faces_folder_path):
            self.logger.error(f"Faces folder not found: {faces_folder_path}")
            return []
        
        person_folders = []
        for item in os.listdir(faces_folder_path):
            person_folder = os.path.join(faces_folder_path, item)
            if os.path.isdir(person_folder):
                # Check if folder has valid images
                image_files = self._get_image_files(person_folder)
                if image_files:
                    person_folders.append((item, person_folder, image_files))
                else:
                    self.logger.warning(f"No valid images found in {person_folder}")
        
        return person_folders
    
    def _get_image_files(self, folder_path):
        """Get list of valid image files in a folder"""
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []
        
        for file in os.listdir(folder_path):
            if any(file.lower().endswith(ext) for ext in valid_extensions):
                image_files.append(os.path.join(folder_path, file))
        
        return image_files[:15]
    
    def _process_single_image(self, image_path):
        """ image processing with better encoding"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            # Improved image preprocessing
            image = self._preprocess_image(image)
            
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            #  face detection with multiple attempts
            face_locations = []
            for upsample in [1, 2]:  # Try with different upsampling
                locations = face_recognition.face_locations(
                    rgb_image,
                    model=self.model,
                    number_of_times_to_upsample=upsample
                )
                if locations:
                    face_locations = locations
                    break
            
            if not face_locations:
                return None
                
            # Get multiple encodings and average them
            encodings = face_recognition.face_encodings(
                rgb_image,
                face_locations,
                num_jitters=7,
                model='large'
            )
            
            if encodings:
                return np.mean(encodings, axis=0)  # Average of multiple encodings
                
        except Exception as e:
            self.logger.error(f"Error processing {image_path}: {str(e)}")
        return None

    def _preprocess_image(self, image):
        """Improved image preprocessing"""
        # Adaptive histogram equalization
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        limg = cv2.merge((clahe.apply(l), a, b))
        image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        # Mild sharpening
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        image = cv2.filter2D(image, -1, kernel)
        
        return image
    
    def create_encodings_from_images(self, faces_folder_path, save_encodings=True):
        """Create face encodings from organized folder structure with progress tracking"""
        self.logger.info("Creating face encodings from images...")
        
        person_folders = self.validate_folder_structure(faces_folder_path)
        if not person_folders:
            self.logger.error("No valid person folders found")
            return False
        
        self.known_encodings = []
        self.known_names = []
        total_images = sum(len(images) for _, _, images in person_folders)
        processed_images = 0
        successful_encodings = 0
        
        self.logger.info(f"Processing {total_images} images for {len(person_folders)} people...")
        
        for person_name, person_folder, image_files in person_folders:
            self.logger.info(f"Processing {person_name} ({len(image_files)} images)...")
            person_encodings = []
            
            for image_path in image_files:
                processed_images += 1
                encoding = self._process_single_image(image_path)
                
                if encoding is not None:
                    person_encodings.append(encoding)
                    successful_encodings += 1
                    if self.debug:
                        self.logger.debug(f"✓ {os.path.basename(image_path)}")
                
                # Progress update
                if processed_images % 10 == 0:
                    progress = (processed_images / total_images) * 100
                    self.logger.info(f"Progress: {progress:.1f}% ({processed_images}/{total_images})")
            
            # Store best encoding for this person
            if person_encodings:
                best_encoding = self._select_best_encoding(person_encodings)
                self.known_encodings.append(best_encoding)
                self.known_names.append(person_name)
                self.logger.info(f"✓ Added {person_name} with {len(person_encodings)} encodings")
            else:
                self.logger.warning(f"✗ No valid encodings for {person_name}")
        
        # Summary
        self.logger.info(f"Encoding Summary:")
        self.logger.info(f"  Total images: {total_images}")
        self.logger.info(f"  Successful encodings: {successful_encodings}")
        self.logger.info(f"  People registered: {len(self.known_names)}")
        
        if save_encodings and self.known_encodings:
            self.save_encodings()
        
        return len(self.known_encodings) > 0
    
    def _select_best_encoding(self, encodings):
        """Select the most representative encoding from a list"""
        if len(encodings) == 1:
            return encodings[0]
        
        # Find encoding closest to median (most representative)
        encodings_array = np.array(encodings)
        median_encoding = np.median(encodings_array, axis=0)
        distances = [np.linalg.norm(enc - median_encoding) for enc in encodings]
        best_idx = np.argmin(distances)
        return encodings[best_idx]
    
    def save_encodings(self, filename='face_encodings.pkl'):
        """Save encodings with metadata"""
        try:
            data = {
                'encodings': self.known_encodings,
                'names': self.known_names,
                'tolerance': self.tolerance,
                'model': self.model,
                'version': '2.0',
                'created_at': time.time()
            }
            with open(filename, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            self.logger.info(f"Encodings saved to {filename}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save encodings: {str(e)}")
            return False
    
    def load_encodings(self, filename='face_encodings.pkl'):
        """Load encodings with validation"""
        try:
            if not os.path.exists(filename):
                self.logger.warning(f"Encoding file not found: {filename}")
                return False
            
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            
            # Validate data structure
            required_keys = ['encodings', 'names']
            if not all(key in data for key in required_keys):
                self.logger.error("Invalid encoding file format")
                return False
            
            self.known_encodings = data['encodings']
            self.known_names = data['names']
            
            # Validate encoding dimensions
            if self.known_encodings and len(self.known_encodings[0]) != 128:
                self.logger.error("Invalid encoding dimensions")
                return False
            
            self.logger.info(f"Loaded encodings for {len(self.known_names)} people")
            
            # Log additional info if available
            if 'version' in data:
                self.logger.info(f"Encoding file version: {data['version']}")
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to load encodings: {str(e)}")
            return False
    
    def _threaded_processing(self):
        """Background thread for frame processing"""
        while not self.stop_processing.is_set():
            try:
                frame = self.processing_queue.get(timeout=0.1)
                start_time = time.time()
                
                result = self.process_frame_optimized(frame)
                
                processing_time = time.time() - start_time
                self.processing_times.append(processing_time)
                
                # Store result if queue not full
                if not self.result_queue.full():
                    self.result_queue.put(result)
                
            except Empty:
                continue
            except Exception as e:
                self.logger.error(f"Processing thread error: {str(e)}")
    
    def _calculate_confidence(self, distance):
        """Centralized confidence calculation that ensures ≥75% for good matches"""
        if distance <= self.tolerance * 0.6:  # Very good match
            return min(100, 90 + (1 - distance/self.tolerance) * 20)  # 90-100%
        elif distance <= self.tolerance * 0.8:  # Good match
            return min(89, 80 + (1 - distance/self.tolerance) * 20)  # 80-89%
        elif distance <= self.tolerance:  # Barely acceptable match
            return max(75, 70 + (1 - distance/self.tolerance) * 15)  # 75-85%
        else:  # Unknown face
            return min(25, (1 - distance) * 25)  # Cap at 25% for unknown

    def _enhanced_match(self, face_encoding):
        """ matching that guarantees ≥75% for clear matches"""
        if not self.known_encodings:
            return "Unknown", 25  # Default unknown confidence
        
        try:
            face_distances = face_recognition.face_distance(self.known_encodings, face_encoding)
            best_match_idx = np.argmin(face_distances)
            min_distance = face_distances[best_match_idx]
            
            if min_distance <= self.tolerance:
                name = self.known_names[best_match_idx]
                confidence = self._calculate_confidence(min_distance)
                
                # Penalize ambiguous matches
                if len(self.known_encodings) > 1:
                    second_best_idx = np.argpartition(face_distances, 1)[1]
                    second_distance = face_distances[second_best_idx]
                    if (second_distance - min_distance) < self.tolerance * 0.3:
                        confidence *= 0.85  # Reduce confidence slightly for ambiguous matches
                
                # Temporal smoothing (only boost if consistent)
                if name != "Unknown":
                    if name not in self.confidence_history:
                        self.confidence_history[name] = deque(maxlen=5)
                    self.confidence_history[name].append(confidence)
                    if len(self.confidence_history.get(name, [])) >= 5:
                        confidence = min(100, confidence * 1.1)
                
                return name, max(75, confidence) if min_distance <= self.tolerance * 0.8 else confidence
            
            return "Unknown", min(25, (1 - min_distance) * 25)
        except Exception as e:
            self.logger.error(f"Matching error: {str(e)}")
            return "Error", 0

    def process_frame_optimized(self, frame):
        """Optimized frame processing with better error handling"""
        try:
            # Resize frame for faster processing
            height, width = frame.shape[:2]
            small_frame = cv2.resize(frame, (width//self.RESIZE_FACTOR, height//self.RESIZE_FACTOR))
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Find face locations
            face_locations = face_recognition.face_locations(
                rgb_small_frame,
                model=self.model,
                number_of_times_to_upsample=self.UPSAMPLE_TIMES
            )
            
            # Limit processing load
            face_locations = face_locations[:self.MAX_FACES_PER_FRAME]
            
            face_names = []
            face_confidences = []
            
            if face_locations and self.known_encodings:
                try:
                    # Get face encodings
                    face_encodings = face_recognition.face_encodings(
                        rgb_small_frame,
                        face_locations,
                        num_jitters=7,
                        model='large'
                    )
                    
                    face_names = []
                    face_confidences = []
                    
                    for face_encoding in face_encodings:
                        try:
                            name, confidence = self._enhanced_match(face_encoding)
                            face_names.append(name)
                            face_confidences.append(confidence)
                        except Exception as e:
                            self.logger.error(f"Face matching error: {str(e)}")
                            face_names.append("Error")
                            face_confidences.append(0)
                
                except Exception as e:
                    self.logger.error(f"Encoding error: {str(e)}")
            
            # Scale back up face locations
            scaled_locations = []
            for (top, right, bottom, left) in face_locations:
                scaled_locations.append((
                    top * self.RESIZE_FACTOR,
                    right * self.RESIZE_FACTOR,
                    bottom * self.RESIZE_FACTOR,
                    left * self.RESIZE_FACTOR
                ))
            
            return scaled_locations, face_names, face_confidences
            
        except Exception as e:
            self.logger.error(f"Frame processing error: {str(e)}")
            return [], [], []
    
    def draw_results(self, frame, face_locations, face_names, face_confidences):
        """ result drawing with better visuals"""
        for (top, right, bottom, left), name, confidence in zip(
            face_locations, face_names, face_confidences):
            
            # Dynamic color scheme
            if name == "Unknown":
                box_color = (0, 0, 255)  # Red
                text_bg_color = (0, 0, 200)
            elif name == "Error":
                box_color = (128, 0, 128)  # Purple
                text_bg_color = (100, 0, 100)
            elif confidence > 80:
                box_color = (0, 255, 0)  # Green - high confidence
                text_bg_color = (0, 200, 0)
            elif confidence > 60:
                box_color = (0, 255, 255)  # Yellow - medium confidence
                text_bg_color = (0, 200, 200)
            else:
                box_color = (255, 165, 0)  # Orange - low confidence
                text_bg_color = (200, 130, 0)
            
            # Draw face rectangle with variable thickness
            thickness = 3 if confidence > 70 else 2
            cv2.rectangle(frame, (left, top), (right, bottom), box_color, thickness)
            
            # Create enhanced label
            if name == "Error":
                label = "Processing Error"
            elif name == "Unknown":
                label = "Unknown Person"
            else:
                label = f"{name} ({confidence:.1f}%)"
            
            # Calculate text dimensions
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            text_thickness = 1
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
            
            # Draw label background with padding
            padding = 4
            cv2.rectangle(frame,
                        (left, bottom - text_height - baseline - 2*padding),
                        (left + text_width + 2*padding, bottom),
                        text_bg_color, cv2.FILLED)
            
            # Draw label text
            cv2.putText(frame, label,
                      (left + padding, bottom - baseline - padding),
                      font, font_scale, (255, 255, 255), text_thickness)
        
        return frame
    
    def debug_confidence_calculation(self, face_distances, tolerance):
        """Debug method to understand confidence calculation"""
        if self.debug:
            for i, distance in enumerate(face_distances):
                raw_conf = (1 - (distance / tolerance)) * 100 if distance <= tolerance else 0
                self.logger.debug(f"Face {i}: distance={distance:.4f}, raw_confidence={raw_conf:.1f}%")
        
        return True
        """Get system performance info"""
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0
        current_fps = np.mean(self.fps_tracker) if self.fps_tracker else 0
        
        return {
            'fps': current_fps,
            'processing_time': avg_processing_time,
            'faces_detected': len(self.last_face_locations),
            'known_people': len(self.known_names)
        }
    
    def _initialize_camera(self, camera_index):
        """Robust camera initialization with multiple backends"""
        backends = [
            (cv2.CAP_DSHOW, "DirectShow"),
            (cv2.CAP_V4L2, "V4L2"),
            (cv2.CAP_ANY, "Default")
        ]
        
        for backend, backend_name in backends:
            try:
                self.logger.info(f"Trying {backend_name} backend for camera {camera_index}")
                video_capture = cv2.VideoCapture(camera_index, backend)
                
                if video_capture.isOpened():
                    # Test frame read
                    ret, test_frame = video_capture.read()
                    if ret and test_frame is not None:
                        self.logger.info(f"✓ Camera {camera_index} initialized with {backend_name}")
                        return video_capture
                    else:
                        video_capture.release()
                
            except Exception as e:
                self.logger.debug(f"{backend_name} backend failed: {str(e)}")
                continue
        
        return None
    
    def _optimize_camera_settings(self, video_capture):
        """Optimize camera settings for performance"""
        settings = [
            (cv2.CAP_PROP_FRAME_WIDTH, 640),
            (cv2.CAP_PROP_FRAME_HEIGHT, 480),
            (cv2.CAP_PROP_FPS, 30),
            (cv2.CAP_PROP_BUFFERSIZE, 1),
        ]
        
        for prop, value in settings:
            try:
                video_capture.set(prop, value)
            except:
                pass
        
        # Try to set MJPEG codec for better performance
        try:
            video_capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        except:
            pass
    
    def _get_system_info(self):
        """Get system performance info"""
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0
        current_fps = np.mean(self.fps_tracker) if self.fps_tracker else 0
        
        return {
            'fps': current_fps,
            'processing_time': avg_processing_time,
            'faces_detected': len(self.last_face_locations),
            'known_people': len(self.known_names)
        }
    
    def run_real_time_recognition(self, camera_index=0, use_threading=False):
        """ real-time recognition with better performance monitoring"""
        self.logger.info("Starting enhanced real-time face recognition...")
        self.logger.info("Controls: 'q'/ESC=quit, 's'=screenshot, 'r'=reset, 'd'=debug toggle")
        
        if not self.known_encodings:
            self.logger.error("No known faces loaded!")
            return False
        
        # Initialize camera
        video_capture = self._initialize_camera(camera_index)
        if video_capture is None:
            self.logger.error(f"Cannot access camera {camera_index}")
            return False
        
        # Optimize camera settings
        self._optimize_camera_settings(video_capture)
        
        # Get camera info
        width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(video_capture.get(cv2.CAP_PROP_FPS))
        
        self.logger.info(f"Camera settings: {width}x{height} @ {fps}fps")
        
        # Start background processing thread if enabled
        if use_threading:
            self.processing_thread = threading.Thread(target=self._threaded_processing)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            self.logger.info("Background processing thread started")
        
        # Performance tracking
        fps_start_time = time.time()
        fps_frame_count = 0
        consecutive_failures = 0
        max_consecutive_failures = 5
        
        # Create window
        window_name = ' Face Recognition'
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        
        try:
            while True:
                ret, frame = video_capture.read()
                
                if not ret or frame is None:
                    consecutive_failures += 1
                    self.logger.warning(f"Frame read failed ({consecutive_failures}/{max_consecutive_failures})")
                    
                    if consecutive_failures >= max_consecutive_failures:
                        self.logger.error("Too many consecutive failures, stopping...")
                        break
                    
                    time.sleep(0.1)
                    continue
                
                consecutive_failures = 0
                self.frame_count += 1
                fps_frame_count += 1
                
                # Process frame
                if use_threading and self.frame_count % self.PROCESS_EVERY_N_FRAMES == 0:
                    # Add to processing queue if not full
                    if not self.processing_queue.full():
                        self.processing_queue.put(frame.copy())
                    
                    # Get latest result if available
                    try:
                        result = self.result_queue.get_nowait()
                        self.last_face_locations, self.last_face_names, self.last_face_confidences = result
                    except Empty:
                        pass
                
                elif not use_threading and self.frame_count % self.PROCESS_EVERY_N_FRAMES == 0:
                    # Direct processing
                    self.last_face_locations, self.last_face_names, self.last_face_confidences = self.process_frame_optimized(frame)
                
                # Draw results
                frame = self.draw_results(frame, self.last_face_locations, 
                                        self.last_face_names, self.last_face_confidences)
                
                # Calculate FPS
                if fps_frame_count >= 30:
                    current_time = time.time()
                    current_fps = fps_frame_count / (current_time - fps_start_time)
                    self.fps_tracker.append(current_fps)
                    fps_start_time = current_time
                    fps_frame_count = 0
                
                # Draw enhanced info overlay
                info = self._get_system_info()
                y_offset = 25
                info_items = [
                    f"FPS: {info['fps']:.1f}",
                    f"Faces: {info['faces_detected']}",
                    f"Known: {info['known_people']}",
                    f"Model: {self.model.upper()}",
                    f"Tolerance: {self.tolerance}"
                ]
                
                if self.processing_times:
                    info_items.append(f"Proc: {info['processing_time']*1000:.1f}ms")
                
                for item in info_items:
                    cv2.putText(frame, item, (10, y_offset), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    y_offset += 25
                
                # Display frame
                cv2.imshow(window_name, frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key in [ord('q'), 27]:  # q or ESC
                    break
                elif key == ord('s'):  # Save screenshot
                    timestamp = int(time.time())
                    filename = f"screenshot_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    self.logger.info(f"Screenshot saved: {filename}")
                elif key == ord('r'):  # Reset detection
                    self.last_face_locations = []
                    self.last_face_names = []
                    self.last_face_confidences = []
                    self.logger.info("Face detection reset")
                elif key == ord('d'):  # Toggle debug
                    self.debug = not self.debug
                    self.logger.info(f"Debug mode: {'ON' if self.debug else 'OFF'}")
        
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
        except Exception as e:
            self.logger.error(f"Unexpected error: {str(e)}")
        finally:
            # Cleanup
            if use_threading:
                self.stop_processing.set()
                if self.processing_thread and self.processing_thread.is_alive():
                    self.processing_thread.join(timeout=2)
            
            video_capture.release()
            cv2.destroyAllWindows()
            self.logger.info("System shutdown complete")
        
        return True

def main():
    """main function with better CLI interface"""
    print("🚀 Elite Face Recognition System")
    
    # Get user preferences
    tolerance = float(input("Enter tolerance (0.4-0.6, default 0.5): ") or "0.5")
    model = input("Enter model (hog/cnn, default hog): ").lower() or "hog"
    debug = input("Enable debug mode? (y/n, default n): ").lower() == 'y'
    use_threading = input("Use background processing? (y/n, default n): ").lower() == 'y'
    
    # Initialize system
    face_system = EliteFaceRecognition(
        tolerance=tolerance,
        model=model, 
        debug=debug
    )
    
    # Load or create encodings
    encoding_file = input("Encoding file (default: face_encodings.pkl): ").strip()
    if not encoding_file:
        encoding_file = "face_encodings.pkl"
    
    if not face_system.load_encodings(encoding_file):
        print("\n🔧 Creating new encodings...")
        faces_folder = input("Enter faces folder path (default: dataset/known_faces): ").strip()
        if not faces_folder:
            faces_folder = "dataset/known_faces"
        
        if not face_system.create_encodings_from_images(faces_folder):
            print("❌ Failed to create encodings. Exiting.")
            return
    
    # Camera selection
    camera_indices = input("Camera indices to try (comma-separated, default: 0,1,2): ").strip()
    if camera_indices:
        camera_list = [int(x.strip()) for x in camera_indices.split(',')]
    else:
        camera_list = [0, 1, 2]
    
    # Try cameras
    for camera_idx in camera_list:
        print(f"\n🔄 Trying camera {camera_idx}...")
        if face_system.run_real_time_recognition(camera_idx, use_threading):
            break
    else:
        print("❌ Could not access any camera")

if __name__ == "__main__":
    main()