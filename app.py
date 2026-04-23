import cv2
import os
import threading
import time
import shutil
import socket
import configparser
import requests
from datetime import datetime
from flask import Flask, render_template, Response, jsonify, request, send_from_directory
import numpy as np
import platform

app = Flask(__name__)

class MotionDetector:
    def __init__(self):

        self.config = configparser.ConfigParser()
        self.config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.ini')
        if not os.path.exists(self.config_path):
            raise FileNotFoundError("config.ini not found. Please create it based on config_template.ini")
        self.config.read(self.config_path)
        
        # Configuration
        self.bot_token = self.config['telegram']['bot_token']
        self.chat_id = self.config['telegram']['chat_id']
        self.recordings_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.config['storage']['recordings_path'])
        self.max_usage = int(self.config['storage']['max_usage_percent'])
        self.recording_duration = int(self.config['motion']['recording_duration'])
        self.threshold = int(self.config['motion']['threshold'])
        self.min_area = int(self.config['motion']['min_area'])

        # Telegram alert toggle (persisted in config)
        alerts_val = self.config.get('telegram', 'alerts_enabled', fallback='true')
        self.alerts_enabled = alerts_val.strip().lower() == 'true'

        # Motion zone: normalised floats 0.0–1.0 (x, y, w, h)
        self.motion_zone = {
            'x': float(self.config.get('motion', 'zone_x', fallback='0.0')),
            'y': float(self.config.get('motion', 'zone_y', fallback='0.0')),
            'w': float(self.config.get('motion', 'zone_w', fallback='1.0')),
            'h': float(self.config.get('motion', 'zone_h', fallback='1.0')),
        }
        self.rotation = int(self.config.get('camera', 'rotation', fallback='0'))

        self.frame_width  = int(self.config['camera']['width'])
        self.frame_height = int(self.config['camera']['height'])

        if self.rotation in (90, 270):
            self.frame_width, self.frame_height = self.frame_height, self.frame_width

        # Motion detection variables
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2()
        self.is_recording = False
        self.last_motion_time = 0

        # Runtime port (set by __main__ after find_free_port())
        self.port = None

        # Camera state
        self.camera = None
        self.camera_ready = False
        self._open_camera()  # attempt initial open; sets camera_ready

        os.makedirs(self.recordings_path, exist_ok=True)
        
        self.start_motion_detection()
        self.start_storage_monitor()

    # ------------------------------------------------------------------
    # Camera open / watchdog
    # ------------------------------------------------------------------

    def _open_camera(self):
        """Try to open the camera. Returns True on success."""
        cap = None
        if platform.system() == "Windows":
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        elif platform.system() == "Darwin":
            cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        else:
            cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

        if cap and cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.config['camera']['width']))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.config['camera']['height']))
            cap.set(cv2.CAP_PROP_FPS, int(self.config['camera']['fps']))
            if self.camera is not None:
                self.camera.release()
            self.camera = cap
            self.camera_ready = True
            return True
        else:
            if cap:
                cap.release()
            self.camera_ready = False
            return False

    def _start_camera_watchdog(self):
        """Poll every 5 s until a camera becomes available, then send a restored alert."""
        def watchdog():
            ip = self.get_local_ip()
            print("Camera watchdog started – polling for camera…")
            while not self.camera_ready:
                time.sleep(5)
                if self._open_camera():
                    print("Camera reconnected.")
                    port = self.port or 5000
                    self.send_telegram_message(
                        f"✅ Camera reconnected!\n"
                        f"Host: {self.get_hostname()}\n"
                        f"IP: {ip}\n"
                        f"Web Interface: http://{ip}:{port}",
                        override=True
                    )
        threading.Thread(target=watchdog, daemon=True).start()

    # ------------------------------------------------------------------
    # Config persistence helpers
    # ------------------------------------------------------------------

    def _ensure_section(self, section):
        if not self.config.has_section(section):
            self.config.add_section(section)

    def save_config(self):
        """Write current in-memory config back to config.ini."""
        with open(self.config_path, 'w') as f:
            self.config.write(f)

    def save_motion_zone(self, x, y, w, h):
        """Persist normalised motion zone to config.ini."""
        self.motion_zone = {'x': x, 'y': y, 'w': w, 'h': h}
        self._ensure_section('motion')
        self.config.set('motion', 'zone_x', str(x))
        self.config.set('motion', 'zone_y', str(y))
        self.config.set('motion', 'zone_w', str(w))
        self.config.set('motion', 'zone_h', str(h))
        self.save_config()

    def save_alerts_enabled(self, enabled: bool):
        """Persist alerts toggle to config.ini."""
        self.alerts_enabled = enabled
        self._ensure_section('telegram')
        self.config.set('telegram', 'alerts_enabled', 'true' if enabled else 'false')
        self.save_config()

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_local_ip(self):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            return "Unable to get IP"
        
    def get_hostname(self):
        try:
            return socket.gethostname()
        except:
            return "Unknown"

    def send_telegram_message(self, message, override=False):
        if not self.alerts_enabled and not override:
            return
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            requests.post(url, data={'chat_id': self.chat_id, 'text': message}, timeout=5)
        except Exception as e:
            print(f"Failed to send Telegram message: {e}")

    def send_startup_message(self):
        host = self.get_hostname()
        ip = self.get_local_ip()
        port = self.port or 5000
        if self.camera_ready:
            message = (
                f"🎥 Motion Detector Started\n"
                f"Host: {host}\n"
                f"IP: {ip}\n"
                f"Web Interface: http://{ip}:{port}"
            )
            self.send_telegram_message(message, override=True)
        else:
            message = (
                f"⚠️ Motion Detector Started — NO CAMERA DETECTED\n"
                f"Host: {host}\n"
                f"IP: {ip}\n"
                f"Web Interface: http://{ip}:{port}\n"
                f"Waiting for camera to be connected…"
            )
            self.send_telegram_message(message, override=True)
            self._start_camera_watchdog()

    # ------------------------------------------------------------------
    # Camera / motion
    # ------------------------------------------------------------------

    def _rotate_frame(self, frame):
        if self.rotation == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotation == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        elif self.rotation == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame
    
    def get_frame(self):
        if not self.camera_ready or self.camera is None:
            return None
        ret, frame = self.camera.read()
        if not ret:
            self._handle_camera_disconnect()
            return None
        return self._rotate_frame(frame)

    def _handle_camera_disconnect(self):
        """Called once when a live camera stops responding. Sends alert and starts watchdog."""
        if not self.camera_ready:
            return  # already handled, avoid duplicate alerts
        self.camera_ready = False
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        print("Camera disconnected.")
        ip = self.get_local_ip()
        port = self.port or 5000
        self.send_telegram_message(
            f"🔌 Camera unplugged / lost!\n"
            f"Host: {self.get_hostname()}\n"
            f"IP: {ip}\n"
            f"Web Interface: http://{ip}:{port}\n"
            f"Waiting for camera to be reconnected…",
            override=True
        )
        self._start_camera_watchdog()

    def _zone_roi(self, frame):
        """Crop frame to the current motion zone."""
        h, w = frame.shape[:2]
        zx = int(self.motion_zone['x'] * w)
        zy = int(self.motion_zone['y'] * h)
        zw = int(self.motion_zone['w'] * w)
        zh = int(self.motion_zone['h'] * h)
        zx = max(0, min(zx, w - 1))
        zy = max(0, min(zy, h - 1))
        zw = max(1, min(zw, w - zx))
        zh = max(1, min(zh, h - zy))
        return frame[zy:zy+zh, zx:zx+zw], (zx, zy, zw, zh)

    def detect_motion(self, frame):
        roi, _ = self._zone_roi(frame)
        learning_rate = 0 if self.is_recording else 0.05
        fg_mask = self.background_subtractor.apply(roi, learningRate=learning_rate)
        _, fg_mask = cv2.threshold(fg_mask, max(140, self.threshold), 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(
            fg_mask,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        )
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c) > self.min_area:
                return True, fg_mask
        return False, fg_mask

    def record_video(self):
        if self.is_recording:
            return
        self.is_recording = True
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.recordings_path, f"motion_{timestamp}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(self.config['camera']['fps'])
        out = cv2.VideoWriter(filename, fourcc, fps, (self.frame_width, self.frame_height))
        start_time = time.time()
        print(f"Recording started: {filename}")
        try:
            while time.time() - start_time < self.recording_duration:
                t0 = time.time()
                frame = self.get_frame()
                if frame is not None:
                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    cv2.putText(frame, ts, (10, self.frame_height - 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    out.write(frame)
                elapsed = time.time() - t0
                time.sleep(max(0, (1.0 / fps) - elapsed))
        finally:
            out.release()
            self.is_recording = False
        print(f"Recording completed: {filename}")
        self.send_telegram_message(f"📹 Motion detected! Recording saved: {timestamp}")

    def start_motion_detection(self):
        def motion_loop():
            while True:
                if not self.camera_ready:
                    time.sleep(1)
                    continue
                frame = self.get_frame()
                if frame is not None:
                    motion_detected, _ = self.detect_motion(frame)
                    if motion_detected and not self.is_recording:
                        current_time = time.time()
                        if current_time - self.last_motion_time > 5:
                            self.last_motion_time = current_time
                            threading.Thread(target=self.record_video, daemon=True).start()
                time.sleep(0.1)
        threading.Thread(target=motion_loop, daemon=True).start()

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def get_storage_usage(self):
        total, used, free = shutil.disk_usage(self.recordings_path)
        return (used / total) * 100

    def cleanup_old_recordings(self):
        recordings = []
        for filename in os.listdir(self.recordings_path):
            if filename.endswith('.mp4'):
                fp = os.path.join(self.recordings_path, filename)
                recordings.append((fp, os.path.getctime(fp)))
        recordings.sort(key=lambda x: x[1])
        deleted_count = 0
        while self.get_storage_usage() > self.max_usage and recordings:
            filepath, _ = recordings.pop(0)
            try:
                os.remove(filepath)
                deleted_count += 1
                print(f"Deleted old recording: {os.path.basename(filepath)}")
            except Exception as e:
                print(f"Error deleting {filepath}: {e}")
        if deleted_count > 0:
            self.send_telegram_message(f"🗑️ Storage cleanup: {deleted_count} old recordings deleted")

    def start_storage_monitor(self):
        def storage_loop():
            while True:
                if self.get_storage_usage() > self.max_usage:
                    self.cleanup_old_recordings()
                time.sleep(60)
        threading.Thread(target=storage_loop, daemon=True).start()

    # ------------------------------------------------------------------
    # Recording list helper (shared by UI + API)
    # ------------------------------------------------------------------

    def get_sorted_recordings(self):
        """Return list of recording dicts sorted newest-first."""
        recs = []
        for filename in os.listdir(self.recordings_path):
            if filename.endswith('.mp4'):
                filepath = os.path.join(self.recordings_path, filename)
                size = os.path.getsize(filepath)
                created = datetime.fromtimestamp(os.path.getctime(filepath))
                recs.append({
                    'filename': filename,
                    'size_mb': round(size / (1024 * 1024), 2),
                    'created': created.strftime('%Y-%m-%d %H:%M:%S'),
                    'created_ts': os.path.getctime(filepath),
                })
        recs.sort(key=lambda x: x['created_ts'], reverse=True)
        # Remove internal sort key before returning
        for r in recs:
            del r['created_ts']
        return recs

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def generate_frames(self):
        while True:
            frame = self.get_frame()
            if frame is None:
                time.sleep(0.1)
                continue

            motion_detected, _ = self.detect_motion(frame)

            fh, fw = frame.shape[:2]
            zx = int(self.motion_zone['x'] * fw)
            zy = int(self.motion_zone['y'] * fh)
            zw = int(self.motion_zone['w'] * fw)
            zh = int(self.motion_zone['h'] * fh)
            cv2.rectangle(frame, (zx, zy), (zx + zw, zy + zh), (255, 165, 0), 2)

            status = "RECORDING" if self.is_recording else "MONITORING"
            color = (0, 0, 255) if self.is_recording else (0, 255, 0)
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            if motion_detected:
                cv2.putText(frame, "MOTION DETECTED", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            alert_text = "ALERTS: ON" if self.alerts_enabled else "ALERTS: OFF"
            alert_color = (0, 255, 100) if self.alerts_enabled else (0, 100, 255)
            cv2.putText(frame, alert_text, (10, fh - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, alert_color, 2)

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, timestamp, (10, fh - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
            if not ret:
                time.sleep(0.05)
                continue

            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n'
                   b'Content-Length: ' + str(len(frame_bytes)).encode() + b'\r\n'
                   b'\r\n' +
                   frame_bytes +
                   b'\r\n')
            time.sleep(0.05)

    def get_jpg_frame(self):
        frame = self.get_frame()
        if frame is None:
            return None

        motion_detected, _ = self.detect_motion(frame)
        fh, fw = frame.shape[:2]
        zx = int(self.motion_zone['x'] * fw)
        zy = int(self.motion_zone['y'] * fh)
        zw = int(self.motion_zone['w'] * fw)
        zh = int(self.motion_zone['h'] * fh)
        cv2.rectangle(frame, (zx, zy), (zx + zw, zy + zh), (255, 165, 0), 2)

        status = "RECORDING" if self.is_recording else "MONITORING"
        color = (0, 0, 255) if self.is_recording else (0, 255, 0)
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        if motion_detected:
            cv2.putText(frame, "MOTION DETECTED", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        alert_text = "ALERTS: ON" if self.alerts_enabled else "ALERTS: OFF"
        alert_color = (0, 255, 100) if self.alerts_enabled else (0, 100, 255)
        cv2.putText(frame, alert_text, (10, fh - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, alert_color, 2)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, fh - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        return buffer.tobytes() if ret else None


# Initialize
detector = MotionDetector()
RECORDINGS_DIR = detector.recordings_path

# ------------------------------------------------------------------
# Path safety helper
# ------------------------------------------------------------------

def safe_path(filename):
    path = os.path.abspath(os.path.join(RECORDINGS_DIR, filename))
    if not path.startswith(os.path.abspath(RECORDINGS_DIR)):
        raise ValueError("Invalid path")
    return path


# ==================================================================
# Routes – UI pages
# ==================================================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/telegram')
def telegram_page():
    return render_template('telegram.html')


# ==================================================================
# Routes – Live feed
# ==================================================================

@app.route('/video_feed')
def video_feed():
    return Response(
        detector.generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame',
        headers={
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            'Pragma': 'no-cache',
            'Expires': '0',
        }
    )

@app.route('/snapshot')
def snapshot():
    frame = detector.get_jpg_frame()
    if frame is None:
        return "No frame", 500
    return Response(frame, mimetype='image/jpeg', headers={
        'Cache-Control': 'no-cache, no-store, must-revalidate',
        'Pragma': 'no-cache',
        'Expires': '0',
    })


# ==================================================================
# Routes – Status / settings
# ==================================================================

@app.route('/status')
def status():
    return jsonify({
        'is_recording': detector.is_recording,
        'storage_usage': round(detector.get_storage_usage(), 2),
        'recordings_count': len([f for f in os.listdir(detector.recordings_path) if f.endswith('.mp4')]),
        'alerts_enabled': detector.alerts_enabled,
        'motion_zone': detector.motion_zone,
        'camera_ready': detector.camera_ready,
    })

@app.route('/settings')
def get_settings():
    return jsonify({
        'threshold': detector.threshold,
        'min_area': detector.min_area,
        'recording_duration': detector.recording_duration,
        'rotation': detector.rotation,
    })

@app.route('/network_info')
def network_info():
    return jsonify({
        'hostname': detector.get_hostname(),
        'ip': detector.get_local_ip(),
    })

@app.route('/save_zone', methods=['POST'])
def save_zone():
    data = request.get_json()
    try:
        x = float(data['x'])
        y = float(data['y'])
        w = float(data['w'])
        h = float(data['h'])
        if not (0 <= x < 1 and 0 <= y < 1 and 0 < w <= 1 and 0 < h <= 1):
            raise ValueError("Values out of range")
        detector.save_motion_zone(x, y, w, h)
        return jsonify({'success': True, 'motion_zone': detector.motion_zone})
    except (KeyError, ValueError, TypeError) as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/toggle_alerts', methods=['POST'])
def toggle_alerts():
    data = request.get_json(silent=True) or {}
    if 'enabled' in data:
        new_state = bool(data['enabled'])
    else:
        new_state = not detector.alerts_enabled
    detector.save_alerts_enabled(new_state)
    return jsonify({'success': True, 'alerts_enabled': detector.alerts_enabled})

@app.route('/update_settings', methods=['POST'])
def update_settings():
    data = request.get_json()
    errors = {}
    updated = {}

    if 'threshold' in data:
        try:
            val = int(data['threshold'])
            if not (1 <= val <= 255):
                raise ValueError("Must be between 1 and 255")
            detector.threshold = val
            detector._ensure_section('motion')
            detector.config.set('motion', 'threshold', str(val))
            updated['threshold'] = val
        except (ValueError, TypeError) as e:
            errors['threshold'] = str(e)

    if 'min_area' in data:
        try:
            val = int(data['min_area'])
            if not (1 <= val <= 100000):
                raise ValueError("Must be between 1 and 100000")
            detector.min_area = val
            detector._ensure_section('motion')
            detector.config.set('motion', 'min_area', str(val))
            updated['min_area'] = val
        except (ValueError, TypeError) as e:
            errors['min_area'] = str(e)

    if 'recording_duration' in data:
        try:
            val = int(data['recording_duration'])
            if not (5 <= val <= 300):
                raise ValueError("Must be between 5 and 300 seconds")
            detector.recording_duration = val
            detector._ensure_section('motion')
            detector.config.set('motion', 'recording_duration', str(val))
            updated['recording_duration'] = val
        except (ValueError, TypeError) as e:
            errors['recording_duration'] = str(e)

    if updated:
        detector.save_config()

    if errors:
        return jsonify({'success': False, 'errors': errors, 'updated': updated}), 400
    return jsonify({'success': True, 'updated': updated})

@app.route('/set_rotation', methods=['POST'])
def set_rotation():
    data = request.get_json()
    try:
        val = int(data['rotation'])
        if val not in (0, 90, 180, 270):
            raise ValueError("Must be 0, 90, 180, or 270")
        detector.rotation = val
        detector._ensure_section('camera')
        detector.config.set('camera', 'rotation', str(val))
        detector.save_config()
        base_w = int(detector.config['camera']['width'])
        base_h = int(detector.config['camera']['height'])
        if val in (90, 270):
            detector.frame_width, detector.frame_height = base_h, base_w
        else:
            detector.frame_width, detector.frame_height = base_w, base_h
        return jsonify({'success': True, 'rotation': val})
    except (KeyError, ValueError, TypeError) as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/update_telegram', methods=['POST'])
def update_telegram():
    data = request.get_json()
    if not data:
        return jsonify({'success': False, 'error': 'No data provided'}), 400

    updated = {}
    errors = {}

    if 'bot_token' in data:
        val = str(data['bot_token']).strip()
        if not val:
            errors['bot_token'] = 'Cannot be empty'
        else:
            detector.bot_token = val
            detector._ensure_section('telegram')
            detector.config.set('telegram', 'bot_token', val)
            updated['bot_token'] = True

    if 'chat_id' in data:
        val = str(data['chat_id']).strip()
        if not val:
            errors['chat_id'] = 'Cannot be empty'
        else:
            detector.chat_id = val
            detector._ensure_section('telegram')
            detector.config.set('telegram', 'chat_id', val)
            updated['chat_id'] = val

    if updated:
        detector.save_config()

    if errors:
        return jsonify({'success': False, 'errors': errors, 'updated': updated}), 400
    if not updated:
        return jsonify({'success': False, 'error': 'No valid fields provided'}), 400
    return jsonify({'success': True, 'updated': updated})

@app.route('/test_telegram', methods=['POST'])
def test_telegram():
    try:
        detector.send_telegram_message("🔔 Test alert from Motion Detector", override=True)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ==================================================================
# Routes – Recordings (UI + streaming)
# ==================================================================

@app.route('/recordings')
def recordings():
    """Full list, newest first. Includes index field for prev/next navigation."""
    recs = detector.get_sorted_recordings()
    for i, r in enumerate(recs):
        r['index'] = i          # 0 = newest
    return jsonify(recs)

@app.route('/recordings/<filename>')
def view_recording(filename):
    path = safe_path(filename)
    return send_from_directory(
        RECORDINGS_DIR,
        os.path.basename(path),
        mimetype='video/mp4',
        conditional=True,
        as_attachment=False,
    )

@app.route('/download/<filename>')
def download_recording(filename):
    try:
        path = safe_path(filename)
        return send_from_directory(RECORDINGS_DIR, os.path.basename(path), as_attachment=True)
    except Exception:
        return "Invalid file", 400


# ==================================================================
# API – Remote recording browser  (JSON, paginated, with neighbour links)
# ==================================================================

@app.route('/api/recordings')
def api_recordings():
    """
    Paginated recording list for remote clients.

    Query parameters:
        page    – 1-based page number (default: 1)
        per_page – items per page (default: 20, max: 100)

    Response:
    {
        "page": 1,
        "per_page": 20,
        "total": 42,
        "total_pages": 3,
        "recordings": [
            {
                "index": 0,           // 0 = newest
                "filename": "motion_20240101_120000.mp4",
                "size_mb": 12.3,
                "created": "2024-01-01 12:00:00",
                "view_url": "/api/recordings/motion_20240101_120000.mp4/stream",
                "download_url": "/download/motion_20240101_120000.mp4",
                "prev": "motion_...",  // older recording filename or null
                "next": "motion_...",  // newer recording filename or null
            },
            ...
        ]
    }
    """
    try:
        page = max(1, int(request.args.get('page', 1)))
        per_page = min(100, max(1, int(request.args.get('per_page', 20))))
    except (ValueError, TypeError):
        return jsonify({'error': 'Invalid pagination parameters'}), 400

    all_recs = detector.get_sorted_recordings()
    total = len(all_recs)
    total_pages = max(1, (total + per_page - 1) // per_page)
    page = min(page, total_pages)

    start = (page - 1) * per_page
    end = start + per_page
    page_recs = all_recs[start:end]

    result = []
    for local_i, rec in enumerate(page_recs):
        global_i = start + local_i          # index within all_recs (0 = newest)
        prev_filename = all_recs[global_i + 1]['filename'] if global_i + 1 < total else None   # older
        next_filename = all_recs[global_i - 1]['filename'] if global_i - 1 >= 0 else None      # newer

        result.append({
            'index': global_i,
            'filename': rec['filename'],
            'size_mb': rec['size_mb'],
            'created': rec['created'],
            'view_url': f"/api/recordings/{rec['filename']}/stream",
            'download_url': f"/download/{rec['filename']}",
            'prev': prev_filename,
            'next': next_filename,
        })

    return jsonify({
        'page': page,
        'per_page': per_page,
        'total': total,
        'total_pages': total_pages,
        'recordings': result,
    })


@app.route('/api/recordings/<filename>/info')
def api_recording_info(filename):
    """
    Return metadata + neighbour links for a single recording by filename.

    Response:
    {
        "filename": "...",
        "size_mb": 12.3,
        "created": "...",
        "index": 5,
        "total": 42,
        "prev": "<older filename or null>",
        "next": "<newer filename or null>",
        "view_url": "/api/recordings/.../stream",
        "download_url": "/download/..."
    }
    """
    all_recs = detector.get_sorted_recordings()
    idx = next((i for i, r in enumerate(all_recs) if r['filename'] == filename), None)
    if idx is None:
        return jsonify({'error': 'Recording not found'}), 404

    rec = all_recs[idx]
    return jsonify({
        'filename': rec['filename'],
        'size_mb': rec['size_mb'],
        'created': rec['created'],
        'index': idx,
        'total': len(all_recs),
        'prev': all_recs[idx + 1]['filename'] if idx + 1 < len(all_recs) else None,
        'next': all_recs[idx - 1]['filename'] if idx - 1 >= 0 else None,
        'view_url': f"/api/recordings/{rec['filename']}/stream",
        'download_url': f"/download/{rec['filename']}",
    })


@app.route('/api/recordings/<filename>/stream')
def api_stream_recording(filename):
    """
    Stream a recording with HTTP Range support (byte-range requests).
    This allows seek/scrub in remote video players.
    """
    try:
        filepath = safe_path(filename)
    except ValueError:
        return jsonify({'error': 'Invalid filename'}), 400

    if not os.path.isfile(filepath):
        return jsonify({'error': 'Recording not found'}), 404

    file_size = os.path.getsize(filepath)
    range_header = request.headers.get('Range', None)

    if range_header:
        # Parse "bytes=start-end"
        try:
            byte_range = range_header.replace('bytes=', '').split('-')
            start = int(byte_range[0])
            end = int(byte_range[1]) if byte_range[1] else file_size - 1
        except (IndexError, ValueError):
            return jsonify({'error': 'Invalid Range header'}), 416

        end = min(end, file_size - 1)
        chunk_size = end - start + 1

        def generate_range():
            with open(filepath, 'rb') as f:
                f.seek(start)
                remaining = chunk_size
                while remaining > 0:
                    data = f.read(min(65536, remaining))
                    if not data:
                        break
                    remaining -= len(data)
                    yield data

        headers = {
            'Content-Range': f'bytes {start}-{end}/{file_size}',
            'Accept-Ranges': 'bytes',
            'Content-Length': str(chunk_size),
            'Content-Type': 'video/mp4',
            'Cache-Control': 'no-cache',
        }
        return Response(generate_range(), status=206, headers=headers)

    # Full file
    def generate_full():
        with open(filepath, 'rb') as f:
            while True:
                data = f.read(65536)
                if not data:
                    break
                yield data

    headers = {
        'Accept-Ranges': 'bytes',
        'Content-Length': str(file_size),
        'Content-Type': 'video/mp4',
        'Cache-Control': 'no-cache',
    }
    return Response(generate_full(), status=200, headers=headers)


@app.route('/api/recordings/latest')
def api_latest_recording():
    """Shortcut – returns info for the most recent recording."""
    all_recs = detector.get_sorted_recordings()
    if not all_recs:
        return jsonify({'error': 'No recordings found'}), 404
    rec = all_recs[0]
    return jsonify({
        'filename': rec['filename'],
        'size_mb': rec['size_mb'],
        'created': rec['created'],
        'index': 0,
        'total': len(all_recs),
        'prev': all_recs[1]['filename'] if len(all_recs) > 1 else None,
        'next': None,
        'view_url': f"/api/recordings/{rec['filename']}/stream",
        'download_url': f"/download/{rec['filename']}",
    })


# ==================================================================
# Startup
# ==================================================================

def find_free_port(start=5000, end=5100):
    for port in range(start, end):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('', port))
                return port
            except OSError:
                continue
    raise RuntimeError("No free port found in range 5000–5100")

if __name__ == '__main__':
    port = find_free_port()
    if port != 5000:
        print(f"Port 5000 in use, using port {port} instead")
    detector.port = port
    detector.send_startup_message()
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
