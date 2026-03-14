import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np
import sounddevice as sd
import threading
import collections
import time
import gc

# Configuration
CAMERA_WIDTH = 1280  # 720p width - becomes height after 90° rotation (720x1280)
CAMERA_HEIGHT = 720  # 720p height - becomes width after rotation
FRAME_RATE = 150  # 150 FPS for ultra-smooth high-speed capture
BUFFER_SECONDS = 10  # 10 seconds of high-speed footage
BUFFER_SIZE = 1500  # Buffer size for ~10 seconds at 150fps
REPLAY_FPS = 150  # Play back at 150 FPS
REPLAY_START_OFFSET = 1.5  # Seconds of footage to capture before the clap
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
        self.playback_speed = 0.50  # Default to 0.50x for slow motion analysis
        self.frame_accumulator = 0.0  # For fractional frame advancement
        self.audio_running = False
        self.replay_window = None  # Separate window for replay
        self.replay_offset = REPLAY_START_OFFSET  # Dynamic replay offset
        self.post_offset = 0.30  # Seconds of footage to capture after the clap
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
        self.total_frames_buffered = 0  # Exact frame counter for post-clap window calc
        self._fps_t0 = None             # For measuring real capture fps
        self.capture_running = False
        # Latest raw frame from each camera, written by capture threads, read by UI thread
        self._latest_frames = [None, None]

        self.setup_ui()
        self.init_cameras()
        self.start_audio()
        self.start_capture()
        self.update()
    
    # ── Design tokens ───────────────────────────────────────────────────────────
    BG       = '#0C0C0F'
    SURFACE  = '#13131A'
    SURFACE2 = '#1A1A24'
    BORDER   = '#252535'
    ACCENT   = '#3B82F6'   # blue
    ACCENT_H = '#60A5FA'
    SUCCESS  = '#22C55E'   # green
    WARN     = '#F59E0B'   # amber
    DANGER   = '#EF4444'
    TEXT     = '#F1F5F9'
    MUTED    = '#64748B'
    # ────────────────────────────────────────────────────────────────────────────

    def _btn(self, parent, text, cmd, **kw):
        """Flat modern button helper."""
        bg = kw.pop('bg', self.SURFACE2)
        fg = kw.pop('fg', self.TEXT)
        return tk.Button(parent, text=text, command=cmd,
                         bg=bg, fg=fg,
                         activebackground=kw.pop('activebackground', self.BORDER),
                         activeforeground=self.TEXT,
                         relief=tk.FLAT, bd=0,
                         font=kw.pop('font', ('Segoe UI', 10)),
                         cursor='hand2',
                         **kw)

    def setup_ui(self):
        self.root.configure(bg=self.BG)
        self.root.title("SwingSense")

        # ── Header bar ──────────────────────────────────────────────────────────
        header = tk.Frame(self.root, bg=self.SURFACE, height=48)
        header.pack(fill=tk.X)
        header.pack_propagate(False)

        tk.Label(header, text="SwingSense", font=('Segoe UI', 15, 'bold'),
                 bg=self.SURFACE, fg=self.TEXT).pack(side=tk.LEFT, padx=20)

        self.status = tk.Label(header, text="  ● LIVE  ",
                               font=('Segoe UI', 10, 'bold'),
                               bg=self.SUCCESS, fg='#000000',
                               padx=8, pady=2)
        self.status.pack(side=tk.LEFT, padx=10, pady=10)

        # ── Camera feeds ────────────────────────────────────────────────────────
        feed_frame = tk.Frame(self.root, bg=self.BG)
        feed_frame.pack(expand=True, pady=(12, 0))

        self.labels = []
        cam_names = ['CAM A', 'CAM B']
        for i in range(2):
            card = tk.Frame(feed_frame, bg=self.SURFACE,
                            highlightbackground=self.BORDER, highlightthickness=1)
            card.pack(side=tk.LEFT, padx=12)
            tk.Label(card, text=cam_names[i], font=('Segoe UI', 9, 'bold'),
                     bg=self.SURFACE, fg=self.MUTED).pack(pady=(6, 2))
            lbl = tk.Label(card, bg='black')
            lbl.pack(padx=6, pady=(0, 6))
            self.labels.append(lbl)

        # ── Bottom settings strip ────────────────────────────────────────────────
        bar = tk.Frame(self.root, bg=self.SURFACE, height=52)
        bar.pack(fill=tk.X, side=tk.BOTTOM)
        bar.pack_propagate(False)

        slider_kw = dict(orient=tk.HORIZONTAL, length=200,
                         bg=self.SURFACE, fg=self.TEXT,
                         troughcolor=self.BORDER, activebackground=self.ACCENT,
                         highlightthickness=0, bd=0,
                         sliderrelief=tk.FLAT, sliderlength=16, width=6)

        tk.Label(bar, text="Before", font=('Segoe UI', 9),
                 bg=self.SURFACE, fg=self.MUTED).pack(side=tk.LEFT, padx=(20, 4), pady=14)
        self.time_slider = tk.Scale(bar, from_=0.1, to=2.5, resolution=0.1,
                                    command=self.update_replay_offset, **slider_kw)
        self.time_slider.set(REPLAY_START_OFFSET)
        self.time_slider.pack(side=tk.LEFT, pady=8)

        tk.Label(bar, text="After", font=('Segoe UI', 9),
                 bg=self.SURFACE, fg=self.MUTED).pack(side=tk.LEFT, padx=(16, 4), pady=14)
        self.post_slider = tk.Scale(bar, from_=0.0, to=0.6, resolution=0.025,
                                    command=self.update_post_offset, **slider_kw)
        self.post_slider.set(self.post_offset)
        self.post_slider.pack(side=tk.LEFT, pady=8)
    
    def update_replay_offset(self, value):
        self.replay_offset = float(value)

    def update_post_offset(self, value):
        self.post_offset = float(value)
    
    def setup_replay_window(self):
        """Create a new replay window for each swing."""
        self.replay_photo_images = [None, None]

        for attr in ('replay_window', 'control_window'):
            win = getattr(self, attr, None)
            if win is not None:
                try:
                    win.destroy()
                except Exception:
                    pass
                setattr(self, attr, None)

        # ── Window ──────────────────────────────────────────────────────────────
        self.replay_window = tk.Toplevel(self.root)
        self.replay_window.title("SwingSense — Replay")
        self.replay_window.protocol("WM_DELETE_WINDOW", self.close_replay_window)
        self.replay_window.configure(bg=self.BG)
        self.replay_window.state('zoomed')

        # ── Camera feeds ────────────────────────────────────────────────────────
        feed_frame = tk.Frame(self.replay_window, bg=self.BG)
        feed_frame.pack(expand=True, pady=(10, 0))

        self.replay_labels = []
        cam_names = ['CAM A', 'CAM B']
        for i in range(2):
            card = tk.Frame(feed_frame, bg=self.SURFACE,
                            highlightbackground=self.BORDER, highlightthickness=1)
            card.pack(side=tk.LEFT, padx=12)
            tk.Label(card, text=cam_names[i], font=('Segoe UI', 9, 'bold'),
                     bg=self.SURFACE, fg=self.MUTED).pack(pady=(6, 2))
            lbl = tk.Label(card, bg='black')
            lbl.pack(padx=6, pady=(0, 6))
            lbl.bind("<Button-1>", lambda e, cam=i: self.on_replay_click(e, cam))
            self.replay_labels.append(lbl)

        # ── Control strip ────────────────────────────────────────────────────────
        strip = tk.Frame(self.replay_window, bg=self.SURFACE, height=90)
        strip.pack(side=tk.BOTTOM, fill=tk.X)
        strip.pack_propagate(False)
        self.controls = strip

        # Row 1 — scrub bar
        scrub_row = tk.Frame(strip, bg=self.SURFACE)
        scrub_row.pack(fill=tk.X, padx=16, pady=(8, 0))

        self.scrub_bar = tk.Scale(scrub_row, from_=0, to=100, orient=tk.HORIZONTAL,
                                  command=self.on_scrub, showvalue=False,
                                  sliderlength=14, sliderrelief=tk.FLAT,
                                  bg=self.SURFACE, activebackground=self.ACCENT,
                                  troughcolor=self.BORDER, fg=self.TEXT,
                                  width=6, highlightthickness=0, bd=0)
        self.scrub_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.scrub_updating = False

        self.frame_label = tk.Label(scrub_row, text="0", font=('Segoe UI', 9),
                                    width=7, bg=self.SURFACE, fg=self.MUTED, anchor='e')
        self.frame_label.pack(side=tk.LEFT, padx=(6, 0))

        # Row 2 — all buttons
        btn_row = tk.Frame(strip, bg=self.SURFACE)
        btn_row.pack(pady=(4, 8))

        PAD = dict(padx=2)

        # Transport
        self._btn(btn_row, "◀", lambda: self.jump(-1), width=5, padx=8, pady=4).pack(side=tk.LEFT, **PAD)
        self.pause_btn = self._btn(btn_row, "⏸", self.toggle_pause,
                                   bg=self.SUCCESS, fg='#000000', width=5, padx=8, pady=4)
        self.pause_btn.pack(side=tk.LEFT, **PAD)
        self._btn(btn_row, "▶", lambda: self.jump(1), width=5, padx=8, pady=4).pack(side=tk.LEFT, **PAD)

        tk.Frame(btn_row, width=1, bg=self.BORDER).pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=4)

        # Speed
        speeds = [('.25x', 0.25), ('.5x', 0.5), ('.75x', 0.75), ('1x', 1.0)]
        self.speed_buttons = []
        for label, val in speeds:
            active = (val == self.playback_speed)
            b = self._btn(btn_row, label, lambda v=val: self.set_speed(v),
                          bg=self.ACCENT if active else self.SURFACE2,
                          fg=self.TEXT, width=5, padx=6, pady=4)
            b.pack(side=tk.LEFT, **PAD)
            self.speed_buttons.append(b)
        self.quarter_btn, self.half_btn, self.three_quarter_btn, self.normal_btn = self.speed_buttons

        tk.Frame(btn_row, width=1, bg=self.BORDER).pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=4)

        # Loop
        loop_bg = self.SUCCESS if self.loop_enabled else self.DANGER
        self.loop_btn = self._btn(btn_row, "⟳ Loop", self.toggle_loop,
                                  bg=loop_bg, fg='#000000' if self.loop_enabled else self.TEXT,
                                  width=7, padx=6, pady=4)
        self.loop_btn.pack(side=tk.LEFT, **PAD)

        tk.Frame(btn_row, width=1, bg=self.BORDER).pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=4)

        # Drawing tools
        tool_color = self.SURFACE2
        self.plane_cam0_btn = self._btn(btn_row, "Plane A", lambda: self.toggle_swing_plane(0),
                                        bg=tool_color, width=7, padx=4, pady=4)
        self.plane_cam0_btn.pack(side=tk.LEFT, **PAD)
        self.plane_cam1_btn = self._btn(btn_row, "Plane B", lambda: self.toggle_swing_plane(1),
                                        bg=tool_color, width=7, padx=4, pady=4)
        self.plane_cam1_btn.pack(side=tk.LEFT, **PAD)
        self.circle_cam0_btn = self._btn(btn_row, "Circle A", lambda: self.toggle_circle(0),
                                         bg=tool_color, width=7, padx=4, pady=4)
        self.circle_cam0_btn.pack(side=tk.LEFT, **PAD)
        self.circle_cam1_btn = self._btn(btn_row, "Circle B", lambda: self.toggle_circle(1),
                                         bg=tool_color, width=7, padx=4, pady=4)
        self.circle_cam1_btn.pack(side=tk.LEFT, **PAD)
        self.line_cam0_btn = self._btn(btn_row, "Line A", lambda: self.toggle_line(0),
                                       bg=tool_color, width=7, padx=4, pady=4)
        self.line_cam0_btn.pack(side=tk.LEFT, **PAD)
        self.line_cam1_btn = self._btn(btn_row, "Line B", lambda: self.toggle_line(1),
                                       bg=tool_color, width=7, padx=4, pady=4)
        self.line_cam1_btn.pack(side=tk.LEFT, **PAD)

        tk.Frame(btn_row, width=1, bg=self.BORDER).pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=4)

        # Erase
        self._btn(btn_row, "Erase A", lambda: self.erase_drawings(0),
                  bg=self.DANGER, fg=self.TEXT, width=7, padx=4, pady=4).pack(side=tk.LEFT, **PAD)
        self._btn(btn_row, "Erase B", lambda: self.erase_drawings(1),
                  bg=self.DANGER, fg=self.TEXT, width=7, padx=4, pady=4).pack(side=tk.LEFT, **PAD)

        tk.Frame(btn_row, width=1, bg=self.BORDER).pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=4)

        # Save
        self._btn(btn_row, "↓ Save", self.save_swing,
                  bg=self.ACCENT, fg=self.TEXT, width=8, padx=8, pady=4).pack(side=tk.LEFT, **PAD)

        # Reset any in-progress drawing state (but keep completed overlays)
        self.setting_swing_plane = [False, False]
        self.setting_circle = [False, False]
        self.setting_line = [False, False]
        self.replay_photo_images = [None, None]

        # Restore button colours to reflect persisted overlay state
        overlay_btns = [
            (self.swing_plane_enabled, [self.plane_cam0_btn, self.plane_cam1_btn],
             ['Plane A', 'Plane B'], ['✓ Plane A', '✓ Plane B']),
            (self.circle_enabled, [self.circle_cam0_btn, self.circle_cam1_btn],
             ['Circle A', 'Circle B'], ['✓ Circle A', '✓ Circle B']),
            (self.line_enabled, [self.line_cam0_btn, self.line_cam1_btn],
             ['Line A', 'Line B'], ['✓ Line A', '✓ Line B']),
        ]
        for enabled_list, btns, off_labels, on_labels in overlay_btns:
            for idx, (btn, off_lbl, on_lbl) in enumerate(zip(btns, off_labels, on_labels)):
                if enabled_list[idx]:
                    btn.config(bg=self.SUCCESS, fg='#000000', text=on_lbl)
                else:
                    btn.config(bg=self.SURFACE2, fg=self.TEXT, text=off_lbl)

    def save_swing(self):
        """Save the current swing to /Swings — uses the full post-clap window, not just the replay slice."""
        frames_to_save = getattr(self, 'swing_frames_full', None) or self.swing_frames
        if not frames_to_save:
            return

        import os
        from datetime import datetime

        swings_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Swings')
        os.makedirs(swings_dir, exist_ok=True)

        dt = datetime.now()
        hour = dt.hour % 12 or 12
        # ampm = 'AM' if dt.hour < 12 else 'PM'
        timestamp = f"{dt.strftime('%Y-%m-%d')}__{hour}{dt.strftime('%M')}"
        fps = min(60, max(1, self.actual_fps))  # cap at 60fps for broad player compatibility

        sample = frames_to_save[0]
        frame_h, frame_w = sample[0].shape[:2]
        total_w = frame_w * len(sample)

        # Try H.264 first, fall back to MPEG-4, both as .mp4
        codecs = [('avc1', f'swing_{timestamp}.mp4'),
                  ('mp4v', f'swing_{timestamp}.mp4')]
        writer = None
        out_path = None
        for codec, fname in codecs:
            p = os.path.join(swings_dir, fname)
            vw = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*codec), fps, (total_w, frame_h))
            if vw.isOpened():
                writer, out_path = vw, p
                print(f"Using codec {codec}")
                break
            vw.release()
        if writer is None:
            print("ERROR: Could not open any video writer")
            return

        for frame_pair in frames_to_save:
            combined = np.concatenate(frame_pair, axis=1)          # side by side
            bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
            writer.write(bgr)

        writer.release()
        print(f"Saved swing to {out_path}")

        # Brief status flash in the replay window
        if self.replay_window and self.replay_window.winfo_exists():
            self.status.config(text="  ✓ SAVED  ", bg=self.SUCCESS, fg='#000000')
            self.root.after(2000, lambda: self.status.config(
                text="  ● LIVE  ", bg=self.SUCCESS, fg='#000000'))
    
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
    
    def start_capture(self):
        """Start one capture thread per camera plus a pairing thread."""
        self.capture_running = True
        for i in range(len(self.caps)):
            threading.Thread(target=self._cam_reader, args=(i,), daemon=True).start()
        threading.Thread(target=self._frame_pairer, daemon=True).start()

    def _cam_reader(self, cam_index):
        """Read one camera, skipping duplicate frames to prevent CPU spinning."""
        cap = self.caps[cam_index]
        last_ts = -1.0
        while self.capture_running:
            if cap and cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    ts = cap.get(cv2.CAP_PROP_POS_MSEC)
                    if ts != last_ts:  # genuine new frame
                        last_ts = ts
                        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                        self._latest_frames[cam_index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    else:
                        time.sleep(0.001)  # duplicate — yield until next frame arrives
            else:
                time.sleep(0.005)

    def _frame_pairer(self):
        """Pair latest frames from both cameras and push to the buffer."""
        while self.capture_running:
            f0, f1 = self._latest_frames[0], self._latest_frames[1]
            if f0 is not None and f1 is not None:
                self.frame_buffer.append([f0, f1])
                self.total_frames_buffered += 1
                self._latest_frames = [None, None]

                if self.total_frames_buffered == 1:
                    self._fps_t0 = time.perf_counter()
                elif self.total_frames_buffered == 121 and self._fps_t0 is not None:
                    elapsed = time.perf_counter() - self._fps_t0
                    self.actual_fps = min(240, max(1, round(120 / elapsed)))
                    print(f"Measured real capture FPS: {self.actual_fps}")
            else:
                time.sleep(0.001)

    def update(self):
        if self.is_replaying:
            self.show_replay()
        else:
            self.show_live_display()

        self.root.after(16, self.update)  # ~60fps UI refresh

    
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
                self.frame_label.config(text=f"{self.replay_index + 1}")
        
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
            self.pause_btn.config(text="▶", bg=self.SURFACE2, fg=self.TEXT)
        else:
            self.pause_btn.config(text="⏸", bg=self.SUCCESS, fg='#000000')
            self.frame_accumulator = 0.0
    
    def on_scrub(self, value):
        if not self.swing_frames or self.scrub_updating:
            return
        
        self.scrub_updating = True
        
        total_frames = len(self.swing_frames)
        progress = float(value) / 100.0
        self.replay_index = int(progress * total_frames)
        self.replay_index = max(0, min(self.replay_index, total_frames - 1))
        
        self.frame_label.config(text=f"{self.replay_index + 1}")
        
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
                btn.config(bg=self.ACCENT, fg=self.TEXT)
            else:
                btn.config(bg=self.SURFACE2, fg=self.TEXT)
    
    def erase_drawings(self, camera_index):
        lbl = 'A' if camera_index == 0 else 'B'
        self.swing_plane_enabled[camera_index] = False
        self.swing_plane_points[camera_index] = []
        self.setting_swing_plane[camera_index] = False
        self.circle_enabled[camera_index] = False
        self.circle_points[camera_index] = []
        self.setting_circle[camera_index] = False
        self.line_enabled[camera_index] = False
        self.line_points[camera_index] = []
        self.setting_line[camera_index] = False
        # Reset button labels
        plane_btn = self.plane_cam0_btn if camera_index == 0 else self.plane_cam1_btn
        circle_btn = self.circle_cam0_btn if camera_index == 0 else self.circle_cam1_btn
        line_btn = self.line_cam0_btn if camera_index == 0 else self.line_cam1_btn
        plane_btn.config(bg=self.SURFACE2, fg=self.TEXT, text=f"Plane {lbl}")
        circle_btn.config(bg=self.SURFACE2, fg=self.TEXT, text=f"Circle {lbl}")
        line_btn.config(bg=self.SURFACE2, fg=self.TEXT, text=f"Line {lbl}")

    def toggle_loop(self):
        self.loop_enabled = not self.loop_enabled
        if self.loop_enabled:
            self.loop_btn.config(bg=self.SUCCESS, fg='#000000')
        else:
            self.loop_btn.config(bg=self.DANGER, fg=self.TEXT)
    
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
        
        frames_before_clap = max(0, int(self.actual_fps * self.replay_offset))
        frames_after_clap = int(self.actual_fps * self.post_offset)
        print(f"Swing triggered: actual_fps={self.actual_fps}, pre={frames_before_clap}, post={frames_after_clap}, buffer_size={len(self.frame_buffer)}")

        buffer_list = list(self.frame_buffer)
        buffer_len = len(buffer_list)
        start_index = max(0, buffer_len - frames_before_clap)

        pre_clap_frames = [frame for frame in buffer_list[start_index:]]

        clap_frame_count = self.total_frames_buffered

        self.status.config(text="● CAPTURING SWING...", fg='#000000', bg='#F59E0B')

        wait_time_ms = int((self.post_offset + 1) * 1000)

        self.post_clap_collection = {
            'pre_clap_frames': pre_clap_frames,
            'clap_frame_count': clap_frame_count,
            'frames_needed': frames_after_clap,
        }
        
        self.root.after(wait_time_ms, self._finalize_replay)
    
    def _finalize_replay(self):
        if not hasattr(self, 'post_clap_collection'):
            return
        
        try:
            collection = self.post_clap_collection

            frames_elapsed = self.total_frames_buffered - collection['clap_frame_count']
            frames_needed = collection['frames_needed']

            buffer_list = list(self.frame_buffer)
            current_len = len(buffer_list)

            start_of_post_clap = max(0, current_len - frames_elapsed)

            # Replay: exactly post_offset seconds after the clap
            post_clap_frames = buffer_list[start_of_post_clap:start_of_post_clap + frames_needed]
            # Save: everything from the clap to now (full wait window)
            self.swing_frames_full = collection['pre_clap_frames'] + buffer_list[start_of_post_clap:]

            print(f"Post-clap: frames_elapsed={frames_elapsed}, replay={len(post_clap_frames)}, save={len(buffer_list[start_of_post_clap:])} frames")

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
            
            self.status.config(text="  ● SWING CAPTURED  ", bg=self.ACCENT, fg=self.TEXT)
            self.root.after(1200, lambda: self.status.config(
                text="  ● LIVE  ", bg=self.SUCCESS, fg='#000000'))
            
            print(f"🎬 NEW SWING! Replaying {total_frames} frames @ {self.actual_fps}fps ({total_frames/self.actual_fps:.1f}s)")
            print(f"  Pre-clap: {len(collection['pre_clap_frames'])} frames, Post-clap: {len(post_clap_frames)} frames")
        
        except Exception as e:
            print(f"Error in _finalize_replay: {e}")
            if hasattr(self, 'post_clap_collection'):
                delattr(self, 'post_clap_collection')
            self.status.config(text="  ● LIVE  ", bg=self.SUCCESS, fg='#000000')
    
    def toggle_swing_plane(self, camera_index):
        print(f"Toggle swing plane called for camera {camera_index}")
        
        lbl = 'A' if camera_index == 0 else 'B'
        btn = self.plane_cam0_btn if camera_index == 0 else self.plane_cam1_btn
        if self.setting_swing_plane[camera_index]:
            self.setting_swing_plane[camera_index] = False
            self.swing_plane_points[camera_index] = []
            if self.swing_plane_enabled[camera_index]:
                btn.config(bg=self.SUCCESS, fg='#000000', text=f"✓ Plane {lbl}")
            else:
                btn.config(bg=self.SURFACE2, fg=self.TEXT, text=f"Plane {lbl}")
        elif self.swing_plane_enabled[camera_index]:
            self.swing_plane_enabled[camera_index] = False
            self.swing_plane_points[camera_index] = []
            btn.config(bg=self.SURFACE2, fg=self.TEXT, text=f"Plane {lbl}")
        else:
            self.setting_swing_plane[camera_index] = True
            self.swing_plane_points[camera_index] = []
            btn.config(bg=self.WARN, fg='#000000', text=f"Click {lbl}")
    
    def toggle_circle(self, camera_index):
        lbl = 'A' if camera_index == 0 else 'B'
        btn = self.circle_cam0_btn if camera_index == 0 else self.circle_cam1_btn
        if self.setting_circle[camera_index]:
            self.setting_circle[camera_index] = False
            self.circle_points[camera_index] = []
            if self.circle_enabled[camera_index]:
                btn.config(bg=self.SUCCESS, fg='#000000', text=f"✓ Circle {lbl}")
            else:
                btn.config(bg=self.SURFACE2, fg=self.TEXT, text=f"Circle {lbl}")
        elif self.circle_enabled[camera_index]:
            self.circle_enabled[camera_index] = False
            self.circle_points[camera_index] = []
            btn.config(bg=self.SURFACE2, fg=self.TEXT, text=f"Circle {lbl}")
        else:
            self.setting_circle[camera_index] = True
            self.circle_points[camera_index] = []
            btn.config(bg=self.WARN, fg='#000000', text=f"Click {lbl}")
    
    def toggle_line(self, camera_index):
        lbl = 'A' if camera_index == 0 else 'B'
        btn = self.line_cam0_btn if camera_index == 0 else self.line_cam1_btn
        if self.setting_line[camera_index]:
            self.setting_line[camera_index] = False
            self.line_points[camera_index] = []
            if self.line_enabled[camera_index]:
                btn.config(bg=self.SUCCESS, fg='#000000', text=f"✓ Line {lbl}")
            else:
                btn.config(bg=self.SURFACE2, fg=self.TEXT, text=f"Line {lbl}")
        elif self.line_enabled[camera_index]:
            self.line_enabled[camera_index] = False
            self.line_points[camera_index] = []
            btn.config(bg=self.SURFACE2, fg=self.TEXT, text=f"Line {lbl}")
        else:
            self.setting_line[camera_index] = True
            self.line_points[camera_index] = []
            btn.config(bg=self.WARN, fg='#000000', text=f"Click {lbl}")
    
    def on_replay_click(self, event, camera_index):
        x = int(event.x / self.display_scale)
        y = int(event.y / self.display_scale)
        
        # Handle swing plane
        if self.setting_swing_plane[camera_index]:
            self.swing_plane_points[camera_index].append((x, y))
            if len(self.swing_plane_points[camera_index]) == 2:
                self.swing_plane_enabled[camera_index] = True
                self.setting_swing_plane[camera_index] = False
                lbl = 'A' if camera_index == 0 else 'B'
                btn = self.plane_cam0_btn if camera_index == 0 else self.plane_cam1_btn
                btn.config(bg=self.SUCCESS, fg='#000000', text=f"✓ Plane {lbl}")

        # Handle circle
        elif self.setting_circle[camera_index]:
            self.circle_points[camera_index].append((x, y))
            if len(self.circle_points[camera_index]) == 2:
                self.circle_enabled[camera_index] = True
                self.setting_circle[camera_index] = False
                lbl = 'A' if camera_index == 0 else 'B'
                btn = self.circle_cam0_btn if camera_index == 0 else self.circle_cam1_btn
                btn.config(bg=self.SUCCESS, fg='#000000', text=f"✓ Circle {lbl}")

        # Handle line
        elif self.setting_line[camera_index]:
            self.line_points[camera_index].append((x, y))
            if len(self.line_points[camera_index]) == 2:
                self.line_enabled[camera_index] = True
                self.setting_line[camera_index] = False
                lbl = 'A' if camera_index == 0 else 'B'
                btn = self.line_cam0_btn if camera_index == 0 else self.line_cam1_btn
                btn.config(bg=self.SUCCESS, fg='#000000', text=f"✓ Line {lbl}")
    
    def close_replay_window(self):
        self.is_replaying = False
        self.replay_photo_images = [None, None]
        self.swing_frames = []
        self.swing_frames_full = []
        self.replay_index = 0
        self.frame_accumulator = 0.0
        self.is_paused = False
        if self.replay_window:
            try:
                self.replay_window.destroy()
            except Exception:
                pass
            self.replay_window = None
        self.status.config(text="  ● LIVE  ", bg=self.SUCCESS, fg='#000000')
    
    def cleanup(self):
        self.capture_running = False
        self.audio_running = False
        time.sleep(0.1)  # let capture threads exit
        for cap in self.caps:
            if cap:
                cap.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = SwingApp(root)
    root.protocol("WM_DELETE_WINDOW", app.cleanup)
    root.mainloop()
