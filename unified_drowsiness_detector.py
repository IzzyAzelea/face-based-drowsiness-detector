import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import threading
import time

# For alarm sounds and notifications
import pygame  # For playing custom alarm sounds
from pathlib import Path
import winsound  # Windows system sounds (fallback)

# Try to import tkinterdnd2 for drag-and-drop support
try:
    from tkinterdnd2 import TkinterDnD, DND_FILES
    DRAG_DROP_AVAILABLE = True
except ImportError:
    DRAG_DROP_AVAILABLE = False
    print("Note: Install tkinterdnd2 for drag-and-drop support: pip install tkinterdnd2")

class UnifiedDrowsinessDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Complete Drowsiness Detection System")
        self.root.geometry("1200x750")
        self.root.configure(bg='#2b2b2b')
        
        # Enable drag-and-drop if available
        if DRAG_DROP_AVAILABLE:
            self.root.drop_target_register(DND_FILES)
            self.root.dnd_bind('<<Drop>>', self.drop_file)
        
        # Download model if needed
        self.model_path = 'face_landmarker.task'
        if not os.path.exists(self.model_path):
            self.download_model()
        
        # Initialize MediaPipe
        self.init_mediapipe()
        
        # File variables
        self.current_file = None
        self.file_type = None  # 'image', 'video', or 'camera'
        
        # Video variables
        self.video_capture = None
        self.is_playing = False
        self.video_thread = None
        self.total_frames = 0
        self.current_frame_num = 0
        self.fps = 30
        
        # Camera variables
        self.camera = None
        self.camera_active = False
        self.camera_thread = None
        self.alert_threshold = 30  # Frames before alert
        self.alert_cooldown = 5  # Seconds between alerts
        self.last_alert_time = 0
        
        # Statistics
        self.drowsy_frames = 0
        self.drowsy_detections = 0
        
        # Preprocessing option
        self.use_preprocessing = True  # Auto-enhance lighting by default
        
        # Alarm and notification settings
        self.alarm_enabled = True
        self.use_custom_alarm = False
        self.custom_alarm_path = None
        self.alarm_volume = 0.7  # 70% volume
        
        # Initialize pygame mixer for alarm sounds
        try:
            pygame.mixer.init()
            self.pygame_available = True
        except:
            self.pygame_available = False
            print("Warning: pygame not available, using system beep only")
        
        # Window state tracking
        self.window_minimized = False
        self.root.bind('<Unmap>', self.on_minimize)
        self.root.bind('<Map>', self.on_restore)
        
        # Create UI
        self.create_widgets()
        
        # Handle window closing
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def download_model(self):
        """Download the face landmark model"""
        print("Downloading face landmark model...")
        url = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'
        urllib.request.urlretrieve(url, self.model_path)
        print("Model downloaded!")
    
    def init_mediapipe(self):
        """Initialize MediaPipe Face Landmarker"""
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
    
    def create_widgets(self):
        """Create the GUI widgets"""
        # Title
        title = tk.Label(
            self.root,
            text="💤 Complete Drowsiness Detection System",
            font=("Arial", 22, "bold"),
            bg='#2b2b2b',
            fg='#ffffff'
        )
        title.pack(pady=15)
        
        # Subtitle
        subtitle = tk.Label(
            self.root,
            text="Images • Videos • Live Camera",
            font=("Arial", 12),
            bg='#2b2b2b',
            fg='#888888'
        )
        subtitle.pack(pady=(0, 10))
        
        # Button frame
        button_frame = tk.Frame(self.root, bg='#2b2b2b')
        button_frame.pack(pady=10)
        
        # Select File Button
        drop_text = "(Drag & Drop or " if DRAG_DROP_AVAILABLE else "("
        self.select_btn = tk.Button(
            button_frame,
            text=f"📁 Select Image/Video {drop_text}Click)",
            command=self.select_file,
            font=("Arial", 10, "bold"),
            bg='#4CAF50',
            fg='white',
            padx=12,
            pady=8,
            cursor='hand2',
            relief=tk.RAISED,
            borderwidth=3
        )
        self.select_btn.pack(side=tk.LEFT, padx=5)
        
        # Camera Button
        self.camera_btn = tk.Button(
            button_frame,
            text="📷 Start Camera",
            command=self.toggle_camera,
            font=("Arial", 10, "bold"),
            bg='#9C27B0',
            fg='white',
            padx=12,
            pady=8,
            cursor='hand2',
            relief=tk.RAISED,
            borderwidth=3
        )
        self.camera_btn.pack(side=tk.LEFT, padx=5)
        
        # Play/Pause Button (for videos)
        self.play_pause_btn = tk.Button(
            button_frame,
            text="▶️ Play",
            command=self.toggle_play_pause,
            font=("Arial", 10, "bold"),
            bg='#2196F3',
            fg='white',
            padx=12,
            pady=8,
            cursor='hand2',
            relief=tk.RAISED,
            borderwidth=3,
            state=tk.DISABLED
        )
        self.play_pause_btn.pack(side=tk.LEFT, padx=5)
        
        # Analyze Button (for images)
        self.analyze_btn = tk.Button(
            button_frame,
            text="🔍 Analyze",
            command=self.analyze_image,
            font=("Arial", 10, "bold"),
            bg='#FF9800',
            fg='white',
            padx=12,
            pady=8,
            cursor='hand2',
            relief=tk.RAISED,
            borderwidth=3,
            state=tk.DISABLED
        )
        self.analyze_btn.pack(side=tk.LEFT, padx=5)
        
        # Clear Button
        self.clear_btn = tk.Button(
            button_frame,
            text="🗑️ Clear",
            command=self.clear_file,
            font=("Arial", 10, "bold"),
            bg='#f44336',
            fg='white',
            padx=12,
            pady=8,
            cursor='hand2',
            relief=tk.RAISED,
            borderwidth=3,
            state=tk.DISABLED
        )
        self.clear_btn.pack(side=tk.LEFT, padx=5)
        
        # Enhancement toggle
        self.enhance_var = tk.BooleanVar(value=True)
        self.enhance_check = tk.Checkbutton(
            button_frame,
            text="✨ Auto-Enhance Lighting",
            variable=self.enhance_var,
            command=self.toggle_enhancement,
            font=("Arial", 9),
            bg='#2b2b2b',
            fg='#ffffff',
            selectcolor='#1a1a1a',
            activebackground='#2b2b2b',
            activeforeground='#ffffff',
            cursor='hand2'
        )
        self.enhance_check.pack(side=tk.LEFT, padx=15)
        
        # Main content frame
        content_frame = tk.Frame(self.root, bg='#2b2b2b')
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Settings panel (collapsible)
        settings_frame = tk.Frame(self.root, bg='#3b3b3b', relief=tk.RAISED, borderwidth=2)
        settings_frame.pack(fill=tk.X, padx=20, pady=(0, 10))
        
        settings_title = tk.Label(
            settings_frame,
            text="⚙️ Alarm Settings",
            font=("Arial", 11, "bold"),
            bg='#3b3b3b',
            fg='#ffffff'
        )
        settings_title.pack(pady=5)
        
        # Alarm enable checkbox
        self.alarm_var = tk.BooleanVar(value=True)
        alarm_check = tk.Checkbutton(
            settings_frame,
            text="🔔 Enable Alarm Sound",
            variable=self.alarm_var,
            font=("Arial", 9),
            bg='#3b3b3b',
            fg='#ffffff',
            selectcolor='#1a1a1a',
            activebackground='#3b3b3b',
            activeforeground='#ffffff'
        )
        alarm_check.pack(pady=2)
        
        # Custom alarm button
        custom_alarm_btn = tk.Button(
            settings_frame,
            text="📁 Select Custom Alarm Sound (Optional)",
            command=self.select_alarm_sound,
            font=("Arial", 9),
            bg='#4a4a4a',
            fg='#ffffff',
            padx=10,
            pady=5
        )
        custom_alarm_btn.pack(pady=5)
        
        # Alarm file label
        self.alarm_file_label = tk.Label(
            settings_frame,
            text="Using: System Beep",
            font=("Arial", 8),
            bg='#3b3b3b',
            fg='#888888'
        )
        self.alarm_file_label.pack(pady=2)
        
        # Volume slider
        volume_frame = tk.Frame(settings_frame, bg='#3b3b3b')
        volume_frame.pack(pady=5)
        
        tk.Label(
            volume_frame,
            text="🔊 Volume:",
            font=("Arial", 9),
            bg='#3b3b3b',
            fg='#ffffff'
        ).pack(side=tk.LEFT, padx=5)
        
        self.volume_slider = tk.Scale(
            volume_frame,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            bg='#3b3b3b',
            fg='#ffffff',
            highlightthickness=0,
            length=150,
            command=self.update_volume
        )
        self.volume_slider.set(70)
        self.volume_slider.pack(side=tk.LEFT, padx=5)
        
        # Left side - Display
        display_frame = tk.Frame(content_frame, bg='#3b3b3b', relief=tk.SUNKEN, borderwidth=2)
        display_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Drop zone label
        drop_text = "Drag & Drop Image/Video\nor Click 'Select Image/Video'\nor Click 'Start Camera'" if DRAG_DROP_AVAILABLE else "Click 'Select Image/Video' Button\nor\nClick 'Start Camera' for Live Detection"
        self.drop_label = tk.Label(
            display_frame,
            text=drop_text,
            font=("Arial", 13),
            bg='#3b3b3b',
            fg='#888888',
            justify=tk.CENTER
        )
        self.drop_label.pack(expand=True)
        
        # Display canvas
        self.display_canvas = tk.Label(display_frame, bg='#3b3b3b')
        
        # Video progress bar
        self.progress_frame = tk.Frame(display_frame, bg='#3b3b3b')
        
        self.progress_label = tk.Label(
            self.progress_frame,
            text="00:00 / 00:00",
            font=("Arial", 10),
            bg='#3b3b3b',
            fg='#ffffff'
        )
        self.progress_label.pack(pady=5)
        
        # Right side - Results
        results_frame = tk.Frame(content_frame, bg='#3b3b3b', relief=tk.SUNKEN, borderwidth=2, width=350)
        results_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
        results_frame.pack_propagate(False)
        
        # Results title
        results_title = tk.Label(
            results_frame,
            text="📊 Analysis Results",
            font=("Arial", 16, "bold"),
            bg='#3b3b3b',
            fg='#ffffff'
        )
        results_title.pack(pady=10)
        
        # File type indicator
        self.file_type_label = tk.Label(
            results_frame,
            text="Type: --",
            font=("Arial", 10),
            bg='#3b3b3b',
            fg='#888888'
        )
        self.file_type_label.pack(pady=2, padx=10)
        
        # Status display
        self.status_label = tk.Label(
            results_frame,
            text="Status: No File Loaded",
            font=("Arial", 13, "bold"),
            bg='#3b3b3b',
            fg='#888888',
            wraplength=330,
            justify=tk.LEFT
        )
        self.status_label.pack(pady=10, padx=10)
        
        # Score display
        self.score_label = tk.Label(
            results_frame,
            text="Drowsiness Score: --/100",
            font=("Arial", 11),
            bg='#3b3b3b',
            fg='#ffffff',
            wraplength=330,
            justify=tk.LEFT
        )
        self.score_label.pack(pady=5, padx=10)
        
        # Alert label
        self.alert_label = tk.Label(
            results_frame,
            text="",
            font=("Arial", 11, "bold"),
            bg='#3b3b3b',
            fg='#FF0000',
            wraplength=330,
            justify=tk.CENTER
        )
        self.alert_label.pack(pady=5, padx=10)
        
        # Separator
        separator = tk.Frame(results_frame, height=2, bg='#555555')
        separator.pack(fill=tk.X, padx=10, pady=10)
        
        # Measurements
        measurements_title = tk.Label(
            results_frame,
            text="Current Measurements:",
            font=("Arial", 11, "bold"),
            bg='#3b3b3b',
            fg='#ffffff'
        )
        measurements_title.pack(pady=5, padx=10, anchor='w')
        
        self.ear_label = tk.Label(
            results_frame,
            text="Eye Aspect Ratio: --",
            font=("Arial", 9),
            bg='#3b3b3b',
            fg='#cccccc',
            wraplength=330,
            justify=tk.LEFT
        )
        self.ear_label.pack(pady=2, padx=10, anchor='w')
        
        self.mar_label = tk.Label(
            results_frame,
            text="Mouth Aspect Ratio: --",
            font=("Arial", 9),
            bg='#3b3b3b',
            fg='#cccccc',
            wraplength=330,
            justify=tk.LEFT
        )
        self.mar_label.pack(pady=2, padx=10, anchor='w')
        
        # Separator
        separator2 = tk.Frame(results_frame, height=2, bg='#555555')
        separator2.pack(fill=tk.X, padx=10, pady=10)
        
        # Statistics
        stats_title = tk.Label(
            results_frame,
            text="📈 Statistics:",
            font=("Arial", 11, "bold"),
            bg='#3b3b3b',
            fg='#ffffff'
        )
        stats_title.pack(pady=5, padx=10, anchor='w')
        
        self.frames_label = tk.Label(
            results_frame,
            text="Frames Analyzed: 0",
            font=("Arial", 9),
            bg='#3b3b3b',
            fg='#cccccc'
        )
        self.frames_label.pack(pady=2, padx=10, anchor='w')
        
        self.detections_label = tk.Label(
            results_frame,
            text="Drowsy Detections: 0",
            font=("Arial", 9),
            bg='#3b3b3b',
            fg='#cccccc'
        )
        self.detections_label.pack(pady=2, padx=10, anchor='w')
        
        self.percentage_label = tk.Label(
            results_frame,
            text="Drowsy Percentage: 0%",
            font=("Arial", 9),
            bg='#3b3b3b',
            fg='#cccccc'
        )
        self.percentage_label.pack(pady=2, padx=10, anchor='w')
        
        # Indicators
        separator3 = tk.Frame(results_frame, height=2, bg='#555555')
        separator3.pack(fill=tk.X, padx=10, pady=10)
        
        indicators_title = tk.Label(
            results_frame,
            text="⚠️ Detected Indicators:",
            font=("Arial", 11, "bold"),
            bg='#3b3b3b',
            fg='#ffffff'
        )
        indicators_title.pack(pady=5, padx=10, anchor='w')
        
        self.indicators_text = tk.Text(
            results_frame,
            font=("Arial", 8),
            bg='#2b2b2b',
            fg='#ffffff',
            height=6,
            width=35,
            relief=tk.FLAT,
            wrap=tk.WORD
        )
        self.indicators_text.pack(pady=5, padx=10, fill=tk.BOTH, expand=True)
    
    def select_file(self):
        """Open file dialog to select an image or video"""
        file_path = filedialog.askopenfilename(
            title="Select an image or video",
            filetypes=[
                ("All supported", "*.jpg *.jpeg *.png *.bmp *.gif *.mp4 *.avi *.mov *.mkv"),
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"),
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.load_file(file_path)
    
    def drop_file(self, event):
        """Handle drag and drop file"""
        file_path = event.data
        # Remove curly braces if present (Windows drag-drop quirk)
        if file_path.startswith('{') and file_path.endswith('}'):
            file_path = file_path[1:-1]
        self.load_file(file_path)
    
    def load_file(self, file_path):
        """Load and determine file type"""
        try:
            # Stop any playing video first
            self.stop_video()
            
            # Get file extension
            ext = os.path.splitext(file_path)[1].lower()
            
            # Determine file type
            image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
            video_exts = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
            
            if ext in image_exts:
                self.load_image(file_path)
            elif ext in video_exts:
                self.load_video(file_path)
            else:
                messagebox.showerror("Unsupported Format", f"File type {ext} is not supported.\n\nSupported formats:\nImages: {', '.join(image_exts)}\nVideos: {', '.join(video_exts)}")
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {str(e)}")
    
    def load_image(self, file_path):
        """Load an image file"""
        try:
            # Read image
            image = cv2.imread(file_path)
            
            if image is None:
                messagebox.showerror("Error", "Could not load image. Please select a valid image file.")
                return
            
            # Store file info
            self.current_file = file_path
            self.file_type = 'image'
            
            # Reset statistics
            self.total_frames = 1
            self.current_frame_num = 0
            self.drowsy_detections = 0
            self.drowsy_frames = 0
            
            # Display the image
            self.display_image(image)
            
            # Update UI
            self.file_type_label.config(text="Type: Image")
            self.analyze_btn.config(state=tk.NORMAL)
            self.play_pause_btn.config(state=tk.DISABLED)
            self.clear_btn.config(state=tk.NORMAL)
            
            # Reset results
            self.reset_results()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def load_video(self, file_path):
        """Load a video file"""
        try:
            # Try to open video
            self.video_capture = cv2.VideoCapture(file_path)
            
            if not self.video_capture.isOpened():
                messagebox.showerror("Error", "Could not open video file. The file may be corrupted or in an unsupported format.")
                return
            
            # Get video properties
            self.total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.video_capture.get(cv2.CAP_PROP_FPS)
            if self.fps == 0:
                self.fps = 30  # Default fallback
            
            # Store file info
            self.current_file = file_path
            self.file_type = 'video'
            self.current_frame_num = 0
            
            # Reset statistics
            self.drowsy_detections = 0
            self.drowsy_frames = 0
            
            # Read and display first frame
            ret, frame = self.video_capture.read()
            if ret:
                self.display_image(frame)
            
            # Update UI
            duration = self.total_frames / self.fps
            minutes = int(duration // 60)
            seconds = int(duration % 60)
            
            self.file_type_label.config(text=f"Type: Video ({minutes:02d}:{seconds:02d}, {self.total_frames} frames)")
            self.analyze_btn.config(state=tk.DISABLED)
            self.play_pause_btn.config(state=tk.NORMAL, text="▶️ Play")
            self.clear_btn.config(state=tk.NORMAL)
            
            # Show progress bar
            self.progress_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
            self.update_progress(0, duration)
            
            # Reset results
            self.reset_results()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load video: {str(e)}")
    
    def display_image(self, cv_image):
        """Display image in the canvas"""
        try:
            # Hide drop label
            self.drop_label.pack_forget()
            
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            
            # Resize to fit display (max 700x550)
            height, width = rgb_image.shape[:2]
            max_width, max_height = 700, 550
            
            scale = min(max_width / width, max_height / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            rgb_image = cv2.resize(rgb_image, (new_width, new_height))
            
            # Convert to PhotoImage
            pil_image = Image.fromarray(rgb_image)
            photo = ImageTk.PhotoImage(pil_image)
            
            # Update canvas
            self.display_canvas.config(image=photo)
            self.display_canvas.image = photo
            self.display_canvas.pack(expand=True)
            
        except Exception as e:
            print(f"Error displaying image: {e}")
    
    def analyze_image(self):
        """Analyze a single image"""
        if self.current_file is None or self.file_type != 'image':
            return
        
        self.analyze_btn.config(state=tk.DISABLED, text="⏳ Analyzing...")
        self.root.update()
        
        threading.Thread(target=self._analyze_image_thread, daemon=True).start()
    
    def _analyze_image_thread(self):
        """Thread for analyzing image"""
        try:
            # Read image
            image = cv2.imread(self.current_file)
            height, width = image.shape[:2]
            
            # Analyze
            score, status, indicators, left_ear, right_ear, mar = self.analyze_frame(image)
            
            # Draw results
            annotated_image = image.copy()
            self.draw_status_on_frame(annotated_image, status, score)
            
            # Update UI
            self.root.after(0, lambda: self.display_image(annotated_image))
            self.root.after(0, lambda: self.update_results(status, score, left_ear, right_ear, mar, indicators))
            self.root.after(0, lambda: self.analyze_btn.config(state=tk.NORMAL, text="🔍 Analyze Image"))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Analysis failed: {str(e)}"))
            self.root.after(0, lambda: self.analyze_btn.config(state=tk.NORMAL, text="🔍 Analyze Image"))
    
    def toggle_play_pause(self):
        """Toggle video playback"""
        if self.is_playing:
            self.pause_video()
        else:
            self.play_video()
    
    def play_video(self):
        """Start video playback"""
        if self.video_capture is None or self.file_type != 'video':
            return
        
        self.is_playing = True
        self.play_pause_btn.config(text="⏸️ Pause")
        
        self.video_thread = threading.Thread(target=self._play_video_thread, daemon=True)
        self.video_thread.start()
    
    def pause_video(self):
        """Pause video playback"""
        self.is_playing = False
        self.play_pause_btn.config(text="▶️ Play")
    
    def stop_video(self):
        """Stop video playback completely"""
        self.is_playing = False
        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None
    
    def toggle_camera(self):
        """Toggle camera on/off"""
        if self.camera_active:
            self.stop_camera()
        else:
            self.start_camera()
    
    def start_camera(self):
        """Start live camera feed"""
        try:
            # Stop any video first
            self.stop_video()
            
            # Try to open camera
            self.camera = cv2.VideoCapture(0)
            
            if not self.camera.isOpened():
                messagebox.showerror(
                    "Camera Error",
                    "Could not access camera. Please check:\n"
                    "• Camera is connected\n"
                    "• Camera permissions are granted\n"
                    "• No other app is using the camera"
                )
                return
            
            # Set camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # Set mode
            self.file_type = 'camera'
            self.camera_active = True
            
            # Reset statistics
            self.total_frames = 0
            self.current_frame_num = 0
            self.drowsy_detections = 0
            self.drowsy_frames = 0
            self.last_alert_time = 0
            
            # Update UI
            self.file_type_label.config(text="Type: Live Camera")
            self.camera_btn.config(text="⏹️ Stop Camera", bg='#f44336')
            self.select_btn.config(state=tk.DISABLED)
            self.analyze_btn.config(state=tk.DISABLED)
            self.play_pause_btn.config(state=tk.DISABLED)
            self.clear_btn.config(state=tk.DISABLED)
            
            # Hide drop label, show display
            self.drop_label.pack_forget()
            self.display_canvas.pack(expand=True)
            
            # Reset results
            self.reset_results()
            self.status_label.config(text="Status: Camera Active", fg='#00FF00')
            
            # Start camera thread
            self.camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
            self.camera_thread.start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start camera: {str(e)}")
    
    def stop_camera(self):
        """Stop camera feed"""
        self.camera_active = False
        
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        
        # Update UI
        self.file_type = None
        self.file_type_label.config(text="Type: --")
        self.camera_btn.config(text="📷 Start Camera", bg='#9C27B0')
        self.select_btn.config(state=tk.NORMAL)
        self.clear_btn.config(state=tk.DISABLED)
        
        # Show drop label
        self.display_canvas.pack_forget()
        self.drop_label.pack(expand=True)
        
        self.status_label.config(text="Status: Camera Stopped", fg='#888888')
        self.alert_label.config(text="")
    
    def _camera_loop(self):
        """Main camera loop - runs in separate thread"""
        while self.camera_active and self.camera is not None:
            ret, frame = self.camera.read()
            
            if not ret:
                print("Failed to read camera frame")
                break
            
            # Flip for mirror effect
            frame = cv2.flip(frame, 1)
            
            self.current_frame_num += 1
            
            # Analyze frame
            score, status, indicators, left_ear, right_ear, mar = self.analyze_frame(frame)
            
            # Track drowsiness for alerts
            if score >= 65:
                self.drowsy_frames += 1
                if self.drowsy_frames >= self.alert_threshold:
                    self.trigger_camera_alert()
            else:
                self.drowsy_frames = 0
            
            # Track statistics
            if score >= 45:
                self.drowsy_detections += 1
            
            # Draw results
            self.draw_status_on_frame(frame, status, score)
            
            # Update display
            self.root.after(0, lambda f=frame: self.display_image(f))
            self.root.after(0, lambda: self.update_results(status, score, left_ear, right_ear, mar, indicators))
            
            # Control frame rate
            time.sleep(0.03)  # ~30 FPS
    
    def trigger_camera_alert(self):
        """Trigger drowsiness alert for camera"""
        current_time = time.time()
        
        if current_time - self.last_alert_time > self.alert_cooldown:
            self.last_alert_time = current_time
            
            # Visual alert in UI
            self.root.after(0, lambda: self.alert_label.config(text="⚠️ DROWSINESS ALERT! ⚠️"))
            
            # Play alarm sound
            self.root.after(0, self.play_alarm)
            
            # Show system notification
            self.root.after(0, lambda: self.show_notification(
                "Drowsiness Detected!",
                "You appear to be drowsy or falling asleep. Take a break!"
            ))
            
            # Clear visual alert after 3 seconds
            self.root.after(3000, lambda: self.alert_label.config(text=""))
    
    def _play_video_thread(self):
        """Thread for playing video"""
        frame_delay = 1.0 / self.fps
        
        while self.is_playing and self.video_capture is not None:
            start_time = time.time()
            
            ret, frame = self.video_capture.read()
            
            if not ret:
                # End of video
                self.root.after(0, self.on_video_end)
                break
            
            self.current_frame_num += 1
            
            # Analyze frame
            score, status, indicators, left_ear, right_ear, mar = self.analyze_frame(frame)
            
            # Track drowsiness
            if score >= 45:
                self.drowsy_detections += 1
                if score >= 65:
                    self.drowsy_frames += 1
            
            # Draw results
            self.draw_status_on_frame(frame, status, score)
            
            # Update display
            self.root.after(0, lambda f=frame: self.display_image(f))
            self.root.after(0, lambda: self.update_results(status, score, left_ear, right_ear, mar, indicators))
            
            # Update progress
            current_time = self.current_frame_num / self.fps
            total_time = self.total_frames / self.fps
            self.root.after(0, lambda: self.update_progress(current_time, total_time))
            
            # Frame rate control
            elapsed = time.time() - start_time
            sleep_time = max(0, frame_delay - elapsed)
            time.sleep(sleep_time)
    
    def on_video_end(self):
        """Handle video ending"""
        self.is_playing = False
        self.play_pause_btn.config(text="▶️ Restart")
        
        # Show summary
        if self.total_frames > 0:
            percentage = (self.drowsy_detections / self.current_frame_num) * 100
            messagebox.showinfo(
                "Video Analysis Complete",
                f"Analysis Summary:\n\n"
                f"Total Frames: {self.current_frame_num}\n"
                f"Drowsy Frames: {self.drowsy_detections}\n"
                f"Drowsy Percentage: {percentage:.1f}%\n"
                f"Very Drowsy Frames: {self.drowsy_frames}"
            )
        
        # Reset video to beginning
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.current_frame_num = 0
    
    def update_progress(self, current, total):
        """Update video progress display"""
        current_min = int(current // 60)
        current_sec = int(current % 60)
        total_min = int(total // 60)
        total_sec = int(total % 60)
        
        self.progress_label.config(text=f"{current_min:02d}:{current_sec:02d} / {total_min:02d}:{total_sec:02d}")
    
    def preprocess_frame(self, frame):
        """
        Preprocess frame to handle poor lighting conditions
        - Enhances brightness
        - Improves contrast
        - Applies histogram equalization
        """
        # Convert to LAB color space for better processing
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        
        # Split channels
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        
        # Merge channels back
        enhanced_lab = cv2.merge([l_enhanced, a, b])
        
        # Convert back to BGR
        enhanced_frame = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # Additional brightness boost if image is too dark
        brightness = np.mean(l)
        if brightness < 100:  # Image is dark
            # Calculate brightness adjustment
            boost = min(1.5, 100 / max(brightness, 1))
            enhanced_frame = cv2.convertScaleAbs(enhanced_frame, alpha=boost, beta=10)
        
        return enhanced_frame
    
    def toggle_enhancement(self):
        """Toggle image enhancement on/off"""
        self.use_preprocessing = self.enhance_var.get()
        status = "enabled" if self.use_preprocessing else "disabled"
        print(f"Auto-enhance lighting: {status}")
    
    def select_alarm_sound(self):
        """Select custom alarm sound file"""
        file_path = filedialog.askopenfilename(
            title="Select Alarm Sound",
            filetypes=[
                ("Audio files", "*.mp3 *.wav *.ogg"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.custom_alarm_path = file_path
            self.use_custom_alarm = True
            filename = Path(file_path).name
            self.alarm_file_label.config(text=f"Using: {filename}")
            print(f"Custom alarm set: {filename}")
    
    def update_volume(self, value):
        """Update alarm volume"""
        self.alarm_volume = int(value) / 100.0
        if self.pygame_available:
            pygame.mixer.music.set_volume(self.alarm_volume)
    
    def play_alarm(self):
        """Play alarm sound"""
        if not self.alarm_var.get():
            return  # Alarm disabled
        
        try:
            if self.use_custom_alarm and self.custom_alarm_path and self.pygame_available:
                # Play custom alarm sound
                pygame.mixer.music.load(self.custom_alarm_path)
                pygame.mixer.music.set_volume(self.alarm_volume)
                pygame.mixer.music.play()
            elif self.pygame_available:
                # Play default alarm sound (system beep as fallback)
                winsound.Beep(1000, 500)  # 1000Hz for 500ms
            else:
                # Fallback to simple beep
                print("\a")
        except Exception as e:
            print(f"Error playing alarm: {e}")
            print("\a")  # Fallback beep
    
    def show_notification(self, title, message):
        """Show system notification"""
        try:
            # Try to import and use win10toast for Windows notifications
            from win10toast import ToastNotifier
            toaster = ToastNotifier()
            toaster.show_toast(
                title,
                message,
                icon_path=None,
                duration=5,
                threaded=True
            )
        except ImportError:
            # Fallback: Show tkinter toplevel notification
            if self.window_minimized or not self.root.focus_get():
                # Create a temporary toplevel window for notification
                notification = tk.Toplevel(self.root)
                notification.title("⚠️ Drowsiness Alert")
                notification.geometry("350x120")
                notification.configure(bg='#FF0000')
                
                # Make it stay on top and flash
                notification.attributes('-topmost', True)
                notification.lift()
                notification.focus_force()
                
                # Flash the taskbar
                try:
                    notification.attributes('-flash', True)
                except:
                    pass
                
                msg = tk.Label(
                    notification,
                    text=f"⚠️ {title}\n\n{message}",
                    font=("Arial", 12, "bold"),
                    bg='#FF0000',
                    fg='#FFFFFF',
                    wraplength=330,
                    justify=tk.CENTER
                )
                msg.pack(expand=True, pady=10)
                
                close_btn = tk.Button(
                    notification,
                    text="OK",
                    command=notification.destroy,
                    font=("Arial", 10, "bold"),
                    bg='#FFFFFF',
                    fg='#FF0000',
                    padx=20,
                    pady=5
                )
                close_btn.pack(pady=10)
                
                # Auto-close after 10 seconds
                notification.after(10000, notification.destroy)
        except Exception as e:
            print(f"Notification error: {e}")
    
    def on_minimize(self, event=None):
        """Handle window minimize"""
        self.window_minimized = True
        print("Window minimized - detection continues in background")
    
    def on_restore(self, event=None):
        """Handle window restore"""
        self.window_minimized = False
        print("Window restored")
    
    def analyze_frame(self, frame):
        """
        Preprocess frame to handle poor lighting conditions
        - Enhances brightness
        - Improves contrast
        - Applies histogram equalization
        """
        # Convert to LAB color space for better processing
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        
        # Split channels
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        
        # Merge channels back
        enhanced_lab = cv2.merge([l_enhanced, a, b])
        
        # Convert back to BGR
        enhanced_frame = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # Additional brightness boost if image is too dark
        brightness = np.mean(l)
        if brightness < 100:  # Image is dark
            # Calculate brightness adjustment
            boost = min(1.5, 100 / max(brightness, 1))
            enhanced_frame = cv2.convertScaleAbs(enhanced_frame, alpha=boost, beta=10)
        
        return enhanced_frame
    
    def analyze_frame(self, frame):
        """Analyze a single frame for drowsiness"""
        try:
            height, width = frame.shape[:2]
            
            # Preprocess frame to handle poor lighting (if enabled)
            if self.use_preprocessing:
                enhanced_frame = self.preprocess_frame(frame)
            else:
                enhanced_frame = frame
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)
            
            # Create MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            
            # Detect landmarks
            detection_result = self.detector.detect(mp_image)
            
            if detection_result.face_landmarks:
                landmarks = detection_result.face_landmarks[0]
                
                # Landmark indices
                left_eye_landmarks = [33, 160, 159, 133, 145, 144]
                right_eye_landmarks = [362, 385, 386, 263, 374, 373]
                mouth_landmarks = [61, 13, 291, 14]
                
                # Calculate ratios
                left_ear = self.calculate_eye_aspect_ratio(left_eye_landmarks, landmarks)
                right_ear = self.calculate_eye_aspect_ratio(right_eye_landmarks, landmarks)
                mar = self.calculate_mouth_aspect_ratio(mouth_landmarks, landmarks)
                
                # Assess drowsiness
                score, status, indicators = self.assess_drowsiness(left_ear, right_ear, mar)
                
                # Draw landmarks
                self.draw_landmarks(frame, landmarks, left_eye_landmarks, right_eye_landmarks, mouth_landmarks, width, height)
                
                return score, status, indicators, left_ear, right_ear, mar
            else:
                return 0, "No Face", [], 0, 0, 0
                
        except Exception as e:
            print(f"Error analyzing frame: {e}")
            return 0, "Error", [], 0, 0, 0
    
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance"""
        return np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)
    
    def calculate_eye_aspect_ratio(self, eye_landmarks, landmarks):
        """Calculate EAR"""
        p1 = landmarks[eye_landmarks[0]]
        p2 = landmarks[eye_landmarks[1]]
        p3 = landmarks[eye_landmarks[2]]
        p4 = landmarks[eye_landmarks[3]]
        p5 = landmarks[eye_landmarks[4]]
        p6 = landmarks[eye_landmarks[5]]
        
        vertical1 = self.calculate_distance(p2, p6)
        vertical2 = self.calculate_distance(p3, p5)
        horizontal = self.calculate_distance(p1, p4)
        
        if horizontal == 0:
            return 0
        return (vertical1 + vertical2) / (2.0 * horizontal)
    
    def calculate_mouth_aspect_ratio(self, mouth_landmarks, landmarks):
        """Calculate MAR"""
        p1 = landmarks[mouth_landmarks[0]]
        p2 = landmarks[mouth_landmarks[1]]
        p5 = landmarks[mouth_landmarks[2]]
        p8 = landmarks[mouth_landmarks[3]]
        
        vertical = self.calculate_distance(p2, p8)
        horizontal = self.calculate_distance(p1, p5)
        
        if horizontal == 0:
            return 0
        return vertical / horizontal
    
    def assess_drowsiness(self, left_ear, right_ear, mar):
        """Assess drowsiness"""
        indicators = []
        score = 0
        
        avg_ear = (left_ear + right_ear) / 2.0
        
        # EAR thresholds (optimized from MRL Eye Dataset: optimal = 0.343)
        if avg_ear < 0.20:
            score += 70
            indicators.append("Eyes fully closed (asleep)")
        elif avg_ear < 0.293:  # 0.343 - 0.05
            score += 60
            indicators.append("Eyes nearly closed (very drowsy)")
        elif avg_ear < 0.343:  # OPTIMAL threshold from dataset
            score += 40
            indicators.append("Eyes partially closed (drowsy)")
        elif avg_ear < 0.393:  # 0.343 + 0.05
            score += 25
            indicators.append("Eyes slightly closing")
        elif avg_ear < 0.443:  # 0.343 + 0.10
            score += 10
            indicators.append("Eyes narrowing slightly")
        
        # MAR thresholds (optimized from YawDD Dataset: optimal = 0.167)
        if mar > 0.267:  # 0.167 + 0.10
            score += 35
            indicators.append("Wide yawning detected")
        elif mar > 0.167:  # OPTIMAL threshold from dataset
            score += 30
            indicators.append("Yawning detected")
        elif mar > 0.117:  # 0.167 - 0.05
            score += 15
            indicators.append("Mouth opening (possible yawn)")
        
        score = min(score, 100)
        
        if score >= 65:
            status = "Very Drowsy / Asleep"
        elif score >= 45:
            status = "Drowsy"
        elif score >= 25:
            status = "Slightly Drowsy"
        elif score >= 10:
            status = "Mildly Tired"
        else:
            status = "Alert"
        
        return score, status, indicators
    
    def draw_landmarks(self, image, landmarks, left_eye, right_eye, mouth, width, height):
        """Draw landmarks on image"""
        for idx in left_eye + right_eye:
            landmark = landmarks[idx]
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
        
        for idx in mouth:
            landmark = landmarks[idx]
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            cv2.circle(image, (x, y), 2, (255, 0, 0), -1)
    
    def draw_status_on_frame(self, image, status, score):
        """Draw status on frame"""
        cv2.rectangle(image, (10, 10), (350, 80), (0, 0, 0), -1)
        
        if status == "Alert":
            color = (0, 255, 0)
        elif status == "Mildly Tired":
            color = (0, 255, 255)
        elif status == "Slightly Drowsy":
            color = (0, 200, 255)
        elif status == "Drowsy":
            color = (0, 100, 255)
        else:
            color = (0, 0, 255)
        
        cv2.putText(image, f"Status: {status}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(image, f"Score: {score}/100", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def update_results(self, status, score, left_ear, right_ear, mar, indicators):
        """Update results display"""
        # Status
        if status == "Alert":
            color = '#00FF00'
        elif status == "Mildly Tired":
            color = '#FFFF00'
        elif status == "Slightly Drowsy":
            color = '#FFA500'
        elif status == "Drowsy":
            color = '#FF6400'
        else:
            color = '#FF0000'
        
        self.status_label.config(text=f"Status: {status}", fg=color)
        self.score_label.config(text=f"Drowsiness Score: {score}/100")
        
        # Measurements
        avg_ear = (left_ear + right_ear) / 2.0
        self.ear_label.config(text=f"Eye Aspect Ratio: {avg_ear:.3f}")
        self.mar_label.config(text=f"Mouth Aspect Ratio: {mar:.3f}")
        
        # Statistics
        analyzed = self.current_frame_num if self.file_type == 'video' else self.total_frames
        self.frames_label.config(text=f"Frames Analyzed: {analyzed}")
        self.detections_label.config(text=f"Drowsy Detections: {self.drowsy_detections}")
        
        if analyzed > 0:
            percentage = (self.drowsy_detections / analyzed) * 100
            self.percentage_label.config(text=f"Drowsy Percentage: {percentage:.1f}%")
        
        # Indicators
        self.indicators_text.delete(1.0, tk.END)
        if indicators:
            for indicator in indicators:
                self.indicators_text.insert(tk.END, f"• {indicator}\n")
        else:
            self.indicators_text.insert(tk.END, "✓ No drowsiness indicators")
        
        # Alert
        if score >= 70:
            self.alert_label.config(text="⚠️ HIGH DROWSINESS! ⚠️")
        else:
            self.alert_label.config(text="")
    
    def reset_results(self):
        """Reset results display"""
        self.status_label.config(text="Status: Ready to Analyze", fg='#888888')
        self.score_label.config(text="Drowsiness Score: --/100")
        self.ear_label.config(text="Eye Aspect Ratio: --")
        self.mar_label.config(text="Mouth Aspect Ratio: --")
        self.frames_label.config(text="Frames Analyzed: 0")
        self.detections_label.config(text="Drowsy Detections: 0")
        self.percentage_label.config(text="Drowsy Percentage: 0%")
        self.indicators_text.delete(1.0, tk.END)
        self.indicators_text.insert(tk.END, "Waiting for analysis...")
        self.alert_label.config(text="")
    
    def clear_file(self):
        """Clear current file"""
        self.stop_video()
        self.stop_camera()
        
        self.current_file = None
        self.file_type = None
        
        # Hide displays
        self.display_canvas.pack_forget()
        self.progress_frame.pack_forget()
        self.drop_label.pack(expand=True)
        
        # Reset UI
        self.file_type_label.config(text="Type: --")
        self.analyze_btn.config(state=tk.DISABLED)
        self.play_pause_btn.config(state=tk.DISABLED)
        self.clear_btn.config(state=tk.DISABLED)
        self.select_btn.config(state=tk.NORMAL)
        
        self.reset_results()
        self.status_label.config(text="Status: No File Loaded")
    
    def on_closing(self):
        """Handle window closing"""
        self.stop_video()
        self.stop_camera()
        self.root.destroy()

# Main execution
if __name__ == "__main__":
    if DRAG_DROP_AVAILABLE:
        root = TkinterDnD.Tk()
    else:
        root = tk.Tk()
    app = UnifiedDrowsinessDetectorGUI(root)
    root.mainloop()
