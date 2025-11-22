import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np
import sounddevice as sd
import threading
import collections
import gc

# Configuration
CAMERA_WIDTH = 1280  # 720p width - becomes height after 90° rotation (720x1280)
CAMERA_HEIGHT = 720  # 720p height - becomes width after rotation
FRAME_RATE = 150  # 150 FPS for ultra-smooth high-speed capture
BUFFER_SECONDS = 10  # 10 seconds of high-speed footage
BUFFER_SIZE = 1500  # Buffer size for ~10 seconds at 150fps
REPLAY_FPS = 150  # Play back at 150 FPS
REPLAY_START_OFFSET = 0.5  # Start replay 0.5 seconds before the clap to capture full takeaway
FRAMES_TO_CAPTURE = 150  # Total frames to capture for replay (1.5 seconds at 100fps)

# Audio
AUDIO_RATE = 44100
AUDIO_CHUNK = 1024
AUDIO_THRESHOLD = 1500  # Lowered from 2500 for more sensitive detection of quieter chips


class SwingApp:
    def __init__(self, root):
        self.root = root
        root.title("Golf Swing Analyzer - Live View")
        
        # Make main window fullscreen
        root.state('zoomed')
        
        self.caps = [None, None]
        self.frame_buffer = collections.deque(maxlen=BUFFER_SIZE)
        self.swing_frames = []  # Fixed snapshot of frames for current swing replay
        self.is_replaying = False
        self.replay_index = 0
        self.is_paused = False
        self.loop_enabled = True
        self.playback_speed = 0.25  # Default to 0.25x for slow motion analysis
        self.frame_accumulator = 0.0  # For fractional frame advancement
        self.audio_running = False
        self.replay_window = None  # Separate window for replay
        self.replay_offset = REPLAY_START_OFFSET  # Dynamic replay offset
        self.actual_fps = 150  # Will be set based on actual camera FPS
        self.display_scale = .69  # Shared display scale for both live and replay windows (100% full size)
        
        # Swing plane line settings
        self.swing_plane_enabled = [False, False]  # One for each camera
        self.swing_plane_points = [[], []]  # Store points for each camera
        self.setting_swing_plane = [False, False]  # Whether currently setting points
        
        # Circle drawing settings
        self.circle_enabled = [False, False]  # One for each camera
        self.circle_points = [[], []]  # Store center and radius point for each camera
        self.setting_circle = [False, False]  # Whether currently setting circle
        
        # Line drawing settings
        self.line_enabled = [False, False]  # One for each camera
        self.line_points = [[], []]  # Store two points for each camera
        self.setting_line = [False, False]  # Whether currently setting line
        
        # Performance optimization: Pre-allocate PhotoImage objects
        self.photo_images = [None, None]
        self.replay_photo_images = [None, None]
        self.last_displayed_frame = [None, None]
        
        self.setup_ui()
        self.init_cameras()
        self.start_audio()
        self.update()
    
    def setup_ui(self):
        # Dark mode colors
        self.root.configure(bg='#1E1E1E')
        
        # Live view cameras
        frame = tk.Frame(self.root, bg='#1E1E1E')
        frame.pack(pady=10)
        
        self.labels = []
        for i in range(2):
            container = tk.Frame(frame, relief=tk.RIDGE, borderwidth=2, padx=5, pady=5, 
                               bg='#2D2D2D', highlightbackground='#4CAF50', highlightthickness=1)
            container.pack(side=tk.LEFT, padx=10)
            tk.Label(container, text=f"Camera {i}", font=('Arial', 12, 'bold'), 
                    bg='#2D2D2D', fg='#4CAF50').pack()
            label = tk.Label(container, bg='black')
            label.pack()
            self.labels.append(label)
        
        self.status = tk.Label(self.root, text="🎙️ Listening... Live Preview", 
                              font=('Arial', 14, 'bold'), fg='#4CAF50', bg='#1E1E1E')
        self.status.pack(pady=10)
        
        # Time slider for replay offset
        slider_frame = tk.Frame(self.root, bg='#2D2D2D', relief=tk.RIDGE, borderwidth=2)
        slider_frame.pack(pady=10, padx=20)
        tk.Label(slider_frame, text="Pre-clap buffer (seconds):", 
                font=('Arial', 10), bg='#2D2D2D', fg='#FFFFFF').pack(pady=5)
        self.time_slider = tk.Scale(slider_frame, from_=0.1, to=1.0, resolution=0.1,
                                    orient=tk.HORIZONTAL, length=300, 
                                    command=self.update_replay_offset,
                                    bg='#3D3D3D', fg='#FFFFFF', troughcolor='#4CAF50',
                                    highlightbackground='#2D2D2D', highlightthickness=0)
        self.time_slider.set(REPLAY_START_OFFSET)
        self.time_slider.pack(pady=5)
    
    def update_replay_offset(self, value):
        """Update the replay start offset from slider."""
        self.replay_offset = float(value)
    
    def setup_replay_window(self):
        """Create a new replay window for each swing."""
        # Clean up previous PhotoImage objects to prevent memory leaks
        self.replay_photo_images = [None, None]
        
        # Close previous windows if they exist
        if self.replay_window is not None:
            try:
                self.replay_window.destroy()
                self.replay_window = None
            except:
                pass
        
        if hasattr(self, 'control_window') and self.control_window is not None:
            try:
                self.control_window.destroy()
                self.control_window = None
            except:
                pass
        
        # Create single replay window with video and controls - DARK MODE
        self.replay_window = tk.Toplevel(self.root)
        self.replay_window.title("Golf Swing Replay - Dark Mode")
        self.replay_window.protocol("WM_DELETE_WINDOW", self.close_replay_window)
        self.replay_window.configure(bg='#1E1E1E')
        
        # Make window fullscreen
        self.replay_window.state('zoomed')
        
        # Replay cameras
        replay_frame = tk.Frame(self.replay_window, bg='#1E1E1E')
        replay_frame.pack(pady=5)
        
        self.replay_labels = []
        # Using shared display_scale from __init__
        
        for i in range(2):
            container = tk.Frame(replay_frame, relief=tk.RIDGE, borderwidth=2, padx=5, pady=5,
                               bg='#2D2D2D', highlightbackground='#4CAF50', highlightthickness=2)
            container.pack(side=tk.LEFT, padx=10)
            tk.Label(container, text=f"Camera {i}", font=('Arial', 12, 'bold'),
                    bg='#2D2D2D', fg='#4CAF50').pack()
            
            label = tk.Label(container, bg='black')
            label.pack()
            label.bind("<Button-1>", lambda e, cam=i: self.on_replay_click(e, cam))
            self.replay_labels.append(label)
        
        # Controls at bottom
        self.controls = tk.Frame(self.replay_window, bg='#1E1E1E')
        self.controls.pack(side=tk.BOTTOM, pady=10, fill=tk.X, padx=20)
        
        # Scrub bar
        scrub_frame = tk.Frame(self.controls, bg='#1E1E1E')
        scrub_frame.pack(pady=5, padx=10, fill=tk.X)
        
        self.scrub_bar = tk.Scale(scrub_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                                 command=self.on_scrub, showvalue=False, length=940,
                                 sliderlength=20, sliderrelief=tk.FLAT, 
                                 bg='#1E1E1E', activebackground='#FFC107',
                                 troughcolor='#4CAF50', fg='#FFFFFF', width=4,
                                 highlightthickness=0, bd=0)
        self.scrub_bar.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        self.scrub_updating = False
        
        self.frame_label = tk.Label(scrub_frame, text="⚫ 0", font=('Arial', 12, 'bold'), 
                                   width=6, bg='#1E1E1E', fg='#FFC107')
        self.frame_label.pack(side=tk.LEFT, padx=10)
        
        # Buttons - Minimalistic design
        btn_frame = tk.Frame(self.controls, bg='#1E1E1E')
        btn_frame.pack(pady=5)
        
        tk.Button(btn_frame, text="◀", command=lambda: self.jump(-1), width=6,
                 bg='#424242', fg='#FFFFFF', font=('Arial', 11),
                 activebackground='#616161', height=2).pack(side=tk.LEFT, padx=1)
        
        self.pause_btn = tk.Button(btn_frame, text="⏸", command=self.toggle_pause,
                                   width=6, bg='#4CAF50', fg='#FFFFFF', font=('Arial', 11),
                                   activebackground='#66BB6A', height=2)
        self.pause_btn.pack(side=tk.LEFT, padx=1)
        
        tk.Button(btn_frame, text="▶", command=lambda: self.jump(1), width=6,
                 bg='#424242', fg='#FFFFFF', font=('Arial', 11),
                 activebackground='#616161', height=2).pack(side=tk.LEFT, padx=1)
        
        tk.Frame(btn_frame, width=10, bg='#1E1E1E').pack(side=tk.LEFT)
        
        self.quarter_btn = tk.Button(btn_frame, text=".25x", command=lambda: self.set_speed(0.25),
                                    width=6, bg='#4CAF50', fg='#FFFFFF', relief=tk.SUNKEN, 
                                    font=('Arial', 11), activebackground='#66BB6A', height=2)
        self.quarter_btn.pack(side=tk.LEFT, padx=1)
        
        self.half_btn = tk.Button(btn_frame, text=".5x", command=lambda: self.set_speed(0.5),
                                 width=6, bg='#3D3D3D', fg='#FFFFFF', font=('Arial', 11),
                                 activebackground='#616161', height=2)
        self.half_btn.pack(side=tk.LEFT, padx=1)
        
        self.three_quarter_btn = tk.Button(btn_frame, text=".75x", command=lambda: self.set_speed(0.75),
                                           width=6, bg='#3D3D3D', fg='#FFFFFF', font=('Arial', 11),
                                           activebackground='#616161', height=2)
        self.three_quarter_btn.pack(side=tk.LEFT, padx=1)
        
        self.normal_btn = tk.Button(btn_frame, text="1x", command=lambda: self.set_speed(1.0),
                                    width=6, bg='#3D3D3D', fg='#FFFFFF', font=('Arial', 11),
                                    activebackground='#616161', height=2)
        self.normal_btn.pack(side=tk.LEFT, padx=1)
        
        tk.Frame(btn_frame, width=10, bg='#1E1E1E').pack(side=tk.LEFT)
        
        self.loop_btn = tk.Button(btn_frame, text="🔁", command=self.toggle_loop,
                                 width=6, bg='#4CAF50', fg='#FFFFFF', font=('Arial', 11),
                                 activebackground='#66BB6A', height=2)
        self.loop_btn.pack(side=tk.LEFT, padx=1)
        
        tk.Frame(btn_frame, width=10, bg='#1E1E1E').pack(side=tk.LEFT)
        
        self.plane_cam0_btn = tk.Button(btn_frame, text="✏️0", command=lambda: self.toggle_swing_plane(0),
                                        width=6, bg='#3D3D3D', fg='#FFFFFF', font=('Arial', 11),
                                        activebackground='#616161', height=2)
        self.plane_cam0_btn.pack(side=tk.LEFT, padx=1)
        
        self.plane_cam1_btn = tk.Button(btn_frame, text="✏️1", command=lambda: self.toggle_swing_plane(1),
                                        width=6, bg='#3D3D3D', fg='#FFFFFF', font=('Arial', 11),
                                        activebackground='#616161', height=2)
        self.plane_cam1_btn.pack(side=tk.LEFT, padx=1)
        
        self.circle_cam0_btn = tk.Button(btn_frame, text="⭕0", command=lambda: self.toggle_circle(0),
                                         width=6, bg='#3D3D3D', fg='#FFFFFF', font=('Arial', 11),
                                         activebackground='#616161', height=2)
        self.circle_cam0_btn.pack(side=tk.LEFT, padx=1)
        
        self.circle_cam1_btn = tk.Button(btn_frame, text="⭕1", command=lambda: self.toggle_circle(1),
                                         width=6, bg='#3D3D3D', fg='#FFFFFF', font=('Arial', 11),
                                         activebackground='#616161', height=2)
        self.circle_cam1_btn.pack(side=tk.LEFT, padx=1)
        
        self.line_cam0_btn = tk.Button(btn_frame, text="─0", command=lambda: self.toggle_line(0),
                                       width=6, bg='#3D3D3D', fg='#FFFFFF', font=('Arial', 11),
                                       activebackground='#616161', height=2)
        self.line_cam0_btn.pack(side=tk.LEFT, padx=1)
        
        self.line_cam1_btn = tk.Button(btn_frame, text="─1", command=lambda: self.toggle_line(1),
                                       width=6, bg='#3D3D3D', fg='#FFFFFF', font=('Arial', 11),
                                       activebackground='#616161', height=2)
        self.line_cam1_btn.pack(side=tk.LEFT, padx=1)
        
        self.speed_buttons = [self.quarter_btn, self.half_btn, self.three_quarter_btn, self.normal_btn]
        
        # Reset swing plane for new replay
        self.swing_plane_enabled = [False, False]
        self.swing_plane_points = [[], []]
        self.setting_swing_plane = [False, False]
        
        # Reset PhotoImage objects for new window
        self.replay_photo_images = [None, None]
    
    def init_cameras(self):
        for i in range(2):
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
                    cap.set(cv2.CAP_PROP_FPS, FRAME_RATE)
                    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
                    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
                    cap.set(cv2.CAP_PROP_ZOOM, 100)
                    cap.set(cv2.CAP_PROP_PAN, 0)
                    cap.set(cv2.CAP_PROP_TILT, 0)
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                    
                    actual_fps = int(cap.get(cv2.CAP_PROP_FPS))
                    
                    self.caps[i] = cap
                    
                    if i == 0:
                        self.actual_fps = actual_fps
                    
                    print(f"Camera {i} ready: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ {actual_fps}fps")
            except Exception as e:
                print(f"Camera {i} error: {e}")
    
    def start_audio(self):
        self.audio_running = True
        threading.Thread(target=self.audio_loop, daemon=True).start()
    
    def audio_loop(self):
        try:
            print("Audio detection active - Always listening!")
            
            last_trigger_time = 0
            import time
            
            def audio_callback(indata, frames, time_info, status):
                nonlocal last_trigger_time
                if status:
                    print(f"Audio status: {status}")
                
                volume = np.abs(indata).mean() * 32768
                current_time = time.time()
                
                if volume > AUDIO_THRESHOLD and (current_time - last_trigger_time) > 3.0:
                    print(f"NEW SWING DETECTED! Volume: {volume}")
                    last_trigger_time = current_time
                    self.root.after(0, self.start_replay)
            
            with sd.InputStream(channels=1, samplerate=AUDIO_RATE, 
                              blocksize=AUDIO_CHUNK, callback=audio_callback):
                while self.audio_running:
                    sd.sleep(100)
                    
        except Exception as e:
            print(f"Audio error: {e}")
    
    def update(self):
        self.buffer_frames()
        
        if self.is_replaying:
            self.show_replay()
        else:
            self.show_live_display()
        
        delay = int(1000/self.actual_fps)
        self.root.after(delay, self.update)
    
    def buffer_frames(self):
        frames = []
        for i, cap in enumerate(self.caps):
            if cap and cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(rgb_frame)
        
        if len(frames) == 2:
            self.frame_buffer.append(frames)
    
    def show_live_display(self):
        if len(self.frame_buffer) > 0:
            frames = self.frame_buffer[-1]
            self._update_display(frames)
    
    def show_replay(self):
        if not self.swing_frames or not self.replay_window:
            return
        
        try:
            if not self.replay_window.winfo_exists():
                return
        except:
            return
        
        total_frames = len(self.swing_frames)
        
        if self.replay_index < total_frames:
            frames = self.swing_frames[self.replay_index]
            self._update_replay_display(frames)
        
        if hasattr(self, 'scrub_bar') and not self.scrub_updating:
            if total_frames > 0:
                progress = (self.replay_index / total_frames) * 100
                old_command = self.scrub_bar['command']
                self.scrub_bar.config(command='')
                self.scrub_bar.set(progress)
                self.scrub_bar.config(command=old_command)
                self.frame_label.config(text=f"⚫ {self.replay_index + 1}")
        
        if not self.is_paused:
            self.frame_accumulator += self.playback_speed
            
            if self.frame_accumulator >= 1.0:
                frames_to_advance = int(self.frame_accumulator)
                self.frame_accumulator -= frames_to_advance
                self.replay_index += frames_to_advance
                
                if self.replay_index >= total_frames:
                    if self.loop_enabled:
                        self.replay_index = 0
                        self.frame_accumulator = 0.0
                    else:
                        self.replay_index = total_frames - 1
                        self.is_paused = True
                        self.frame_accumulator = 0.0
    
    def toggle_pause(self):
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.pause_btn.config(text="▶", bg='#424242')
        else:
            self.pause_btn.config(text="⏸", bg='#4CAF50')
            self.frame_accumulator = 0.0
    
    def on_scrub(self, value):
        if not self.swing_frames or self.scrub_updating:
            return
        
        self.scrub_updating = True
        
        total_frames = len(self.swing_frames)
        progress = float(value) / 100.0
        self.replay_index = int(progress * total_frames)
        self.replay_index = max(0, min(self.replay_index, total_frames - 1))
        
        self.frame_label.config(text=f"⚫ {self.replay_index + 1}")
        
        self.scrub_updating = False
    
    def jump(self, frames):
        if not self.swing_frames:
            return
        total_frames = len(self.swing_frames)
        self.replay_index += frames
        self.replay_index = max(0, min(self.replay_index, total_frames - 1))
    
    def set_speed(self, speed):
        self.playback_speed = speed
        self.frame_accumulator = 0.0
        
        speeds = [0.25, 0.5, 0.75, 1.0]
        for i, btn in enumerate(self.speed_buttons):
            if speeds[i] == speed:
                btn.config(bg='#4CAF50', relief=tk.SUNKEN)
            else:
                btn.config(bg='#3D3D3D', relief=tk.RAISED)
    
    def toggle_loop(self):
        self.loop_enabled = not self.loop_enabled
        if self.loop_enabled:
            self.loop_btn.config(bg='#4CAF50')
        else:
            self.loop_btn.config(bg='#E53935')
    
    def _update_display(self, frames):
        for i, rgb_frame in enumerate(frames):
            pil_image = Image.fromarray(rgb_frame)
            
            display_width = int(pil_image.width * self.display_scale)
            display_height = int(pil_image.height * self.display_scale)
            pil_image = pil_image.resize((display_width, display_height), Image.Resampling.LANCZOS)
            
            # Create new PhotoImage each time for reliable display
            self.photo_images[i] = ImageTk.PhotoImage(image=pil_image)
            self.labels[i].configure(image=self.photo_images[i])
            self.labels[i].image = self.photo_images[i]  # Keep a reference
    
    def _update_replay_display(self, frames):
        for i, rgb_frame in enumerate(frames):
            frame_with_overlay = rgb_frame.copy()
            
            if self.swing_plane_enabled[i] and len(self.swing_plane_points[i]) == 2:
                pt1, pt2 = self.swing_plane_points[i]
                
                h, w = frame_with_overlay.shape[:2]
                x1, y1 = pt1
                x2, y2 = pt2
                
                if x2 != x1:
                    slope = (y2 - y1) / (x2 - x1)
                    y_left = int(y1 + slope * (0 - x1))
                    y_right = int(y1 + slope * (w - x1))
                    
                    cv2.line(frame_with_overlay, (0, y_left), (w, y_right), (255, 255, 0), 3)
                else:
                    cv2.line(frame_with_overlay, (x1, 0), (x1, h), (255, 255, 0), 3)
                
                cv2.circle(frame_with_overlay, pt1, 8, (255, 0, 0), -1)
                cv2.circle(frame_with_overlay, pt2, 8, (255, 0, 0), -1)
            
            # Draw circle if enabled
            if self.circle_enabled[i] and len(self.circle_points[i]) == 2:
                center = self.circle_points[i][0]
                edge = self.circle_points[i][1]
                radius = int(np.sqrt((edge[0] - center[0])**2 + (edge[1] - center[1])**2))
                cv2.circle(frame_with_overlay, center, radius, (0, 255, 255), 3)
                cv2.circle(frame_with_overlay, center, 5, (0, 255, 0), -1)
            
            # Draw line if enabled
            if self.line_enabled[i] and len(self.line_points[i]) == 2:
                pt1, pt2 = self.line_points[i]
                cv2.line(frame_with_overlay, pt1, pt2, (255, 0, 255), 3)
                cv2.circle(frame_with_overlay, pt1, 5, (255, 0, 0), -1)
                cv2.circle(frame_with_overlay, pt2, 5, (255, 0, 0), -1)
            
            if self.setting_swing_plane[i]:
                points_needed = 2 - len(self.swing_plane_points[i])
                text = f"Click {points_needed} point(s) to set swing plane"
                cv2.putText(frame_with_overlay, text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                if len(self.swing_plane_points[i]) == 1:
                    pt = self.swing_plane_points[i][0]
                    cv2.circle(frame_with_overlay, pt, 8, (255, 0, 0), -1)
            
            if self.setting_circle[i]:
                points_needed = 2 - len(self.circle_points[i])
                text = "Click center" if points_needed == 2 else "Click edge"
                cv2.putText(frame_with_overlay, text, (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                if len(self.circle_points[i]) == 1:
                    cv2.circle(frame_with_overlay, self.circle_points[i][0], 5, (0, 255, 0), -1)
            
            if self.setting_line[i]:
                points_needed = 2 - len(self.line_points[i])
                text = f"Click {points_needed} point(s) for line"
                cv2.putText(frame_with_overlay, text, (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                if len(self.line_points[i]) == 1:
                    cv2.circle(frame_with_overlay, self.line_points[i][0], 5, (255, 0, 0), -1)
            
            pil_image = Image.fromarray(frame_with_overlay)
            
            # Always apply display scale
            display_width = int(pil_image.width * self.display_scale)
            display_height = int(pil_image.height * self.display_scale)
            pil_image = pil_image.resize((display_width, display_height), Image.Resampling.LANCZOS)
            
            # Create new PhotoImage each time for reliable display
            self.replay_photo_images[i] = ImageTk.PhotoImage(image=pil_image)
            self.replay_labels[i].configure(image=self.replay_photo_images[i])
            self.replay_labels[i].image = self.replay_photo_images[i]  # Keep a reference
    
    def start_replay(self):
        if not self.frame_buffer:
            return
        
        if self.replay_window is not None:
            try:
                self.is_replaying = False
                self.replay_window.destroy()
                self.replay_window = None
            except:
                pass
        
        frames_before_clap = int(self.actual_fps * self.replay_offset) - 40
        frames_after_clap = int(self.actual_fps * 0.5)
        
        buffer_list = list(self.frame_buffer)
        buffer_len = len(buffer_list)
        start_index = max(0, buffer_len - frames_before_clap)
        
        pre_clap_frames = [frame for frame in buffer_list[start_index:]]
        
        clap_buffer_position = len(self.frame_buffer)
        
        self.status.config(text=f"🎬 CLAP DETECTED! Capturing...", fg='#000000', bg='#FFC107')
        
        wait_time_ms = int((0.5 * 1000) + 200)
        
        self.post_clap_collection = {
            'pre_clap_frames': pre_clap_frames,
            'start_position': clap_buffer_position,
            'frames_needed': frames_after_clap
        }
        
        self.root.after(wait_time_ms, self._finalize_replay)
    
    def _finalize_replay(self):
        if not hasattr(self, 'post_clap_collection'):
            return
        
        try:
            collection = self.post_clap_collection
            
            buffer_list = list(self.frame_buffer)
            start_pos = collection['start_position']
            frames_needed = collection['frames_needed']
            
            current_len = len(buffer_list)
            
            if start_pos < current_len:
                end_pos = min(start_pos + frames_needed, current_len)
                post_clap_frames = buffer_list[start_pos:end_pos]
            else:
                post_clap_frames = buffer_list[-frames_needed:] if current_len >= frames_needed else buffer_list
            
            print(f"Collecting post-clap frames: start_pos={start_pos}, current_len={current_len}, grabbed {len(post_clap_frames)} frames")
            
            self.swing_frames = collection['pre_clap_frames'] + post_clap_frames
            
            total_frames = len(self.swing_frames)
            
            delattr(self, 'post_clap_collection')
            
            if total_frames == 0:
                print("Warning: No frames captured for replay")
                return
            
            self.setup_replay_window()
            
            self.is_replaying = True
            self.replay_index = 0
            self.is_paused = False
            self.frame_accumulator = 0.0
            
            self.status.config(text=f"🎬 SWING CAPTURED!", fg='#4CAF50', bg='#1E1E1E')
            self.root.after(800, lambda: self.status.config(
                text="🎙️ Listening... Live Preview", 
                fg='#4CAF50', bg='#1E1E1E'))
            
            print(f"🎬 NEW SWING! Replaying {total_frames} frames @ {self.actual_fps}fps ({total_frames/self.actual_fps:.1f}s)")
            print(f"  Pre-clap: {len(collection['pre_clap_frames'])} frames, Post-clap: {len(post_clap_frames)} frames")
        
        except Exception as e:
            print(f"Error in _finalize_replay: {e}")
            if hasattr(self, 'post_clap_collection'):
                delattr(self, 'post_clap_collection')
            self.status.config(text="🎙️ Listening... Live Preview", fg='#4CAF50', bg='#1E1E1E')
    
    def toggle_swing_plane(self, camera_index):
        print(f"Toggle swing plane called for camera {camera_index}")
        
        if self.setting_swing_plane[camera_index]:
            self.setting_swing_plane[camera_index] = False
            self.swing_plane_points[camera_index] = []
            btn = self.plane_cam0_btn if camera_index == 0 else self.plane_cam1_btn
            if self.swing_plane_enabled[camera_index]:
                btn.config(bg='#4CAF50', text="✓ Plane")
            else:
                btn.config(bg='#3D3D3D', text="✏️ Plane")
        elif self.swing_plane_enabled[camera_index]:
            self.swing_plane_enabled[camera_index] = False
            self.swing_plane_points[camera_index] = []
            btn = self.plane_cam0_btn if camera_index == 0 else self.plane_cam1_btn
            btn.config(bg='#3D3D3D', text="✏️ Plane")
        else:
            self.setting_swing_plane[camera_index] = True
            self.swing_plane_points[camera_index] = []
            btn = self.plane_cam0_btn if camera_index == 0 else self.plane_cam1_btn
            btn.config(bg='#FFA726', text="⬇ Click")
    
    def toggle_circle(self, camera_index):
        if self.setting_circle[camera_index]:
            self.setting_circle[camera_index] = False
            self.circle_points[camera_index] = []
            btn = self.circle_cam0_btn if camera_index == 0 else self.circle_cam1_btn
            if self.circle_enabled[camera_index]:
                btn.config(bg='#4CAF50', text="✓ Circle")
            else:
                btn.config(bg='#3D3D3D', text="⭕ Circle")
        elif self.circle_enabled[camera_index]:
            self.circle_enabled[camera_index] = False
            self.circle_points[camera_index] = []
            btn = self.circle_cam0_btn if camera_index == 0 else self.circle_cam1_btn
            btn.config(bg='#3D3D3D', text="⭕ Circle")
        else:
            self.setting_circle[camera_index] = True
            self.circle_points[camera_index] = []
            btn = self.circle_cam0_btn if camera_index == 0 else self.circle_cam1_btn
            btn.config(bg='#FFA726', text="⬇ Click")
    
    def toggle_line(self, camera_index):
        if self.setting_line[camera_index]:
            self.setting_line[camera_index] = False
            self.line_points[camera_index] = []
            btn = self.line_cam0_btn if camera_index == 0 else self.line_cam1_btn
            if self.line_enabled[camera_index]:
                btn.config(bg='#4CAF50', text="✓ Line")
            else:
                btn.config(bg='#3D3D3D', text="─ Line")
        elif self.line_enabled[camera_index]:
            self.line_enabled[camera_index] = False
            self.line_points[camera_index] = []
            btn = self.line_cam0_btn if camera_index == 0 else self.line_cam1_btn
            btn.config(bg='#3D3D3D', text="─ Line")
        else:
            self.setting_line[camera_index] = True
            self.line_points[camera_index] = []
            btn = self.line_cam0_btn if camera_index == 0 else self.line_cam1_btn
            btn.config(bg='#FFA726', text="⬇ Click")
    
    def on_replay_click(self, event, camera_index):
        x = int(event.x / self.display_scale)
        y = int(event.y / self.display_scale)
        
        # Handle swing plane
        if self.setting_swing_plane[camera_index]:
            self.swing_plane_points[camera_index].append((x, y))
            if len(self.swing_plane_points[camera_index]) == 2:
                self.swing_plane_enabled[camera_index] = True
                self.setting_swing_plane[camera_index] = False
                btn = self.plane_cam0_btn if camera_index == 0 else self.plane_cam1_btn
                btn.config(bg='#4CAF50', text="✓ Plane")
        
        # Handle circle
        elif self.setting_circle[camera_index]:
            self.circle_points[camera_index].append((x, y))
            if len(self.circle_points[camera_index]) == 2:
                self.circle_enabled[camera_index] = True
                self.setting_circle[camera_index] = False
                btn = self.circle_cam0_btn if camera_index == 0 else self.circle_cam1_btn
                btn.config(bg='#4CAF50', text="✓ Circle")
        
        # Handle line
        elif self.setting_line[camera_index]:
            self.line_points[camera_index].append((x, y))
            if len(self.line_points[camera_index]) == 2:
                self.line_enabled[camera_index] = True
                self.setting_line[camera_index] = False
                btn = self.line_cam0_btn if camera_index == 0 else self.line_cam1_btn
                btn.config(bg='#4CAF50', text="✓ Line")
    
    def close_replay_window(self):
        self.is_replaying = False
        self.replay_photo_images = [None, None]
        self.swing_frames = []
        if self.replay_window:
            try:
                self.replay_window.destroy()
            except:
                pass
            self.replay_window = None
    
    def cleanup(self):
        self.audio_running = False
        for cap in self.caps:
            if cap:
                cap.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = SwingApp(root)
    root.protocol("WM_DELETE_WINDOW", app.cleanup)
    root.mainloop()
