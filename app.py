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
        self.recordings_path = self.config['storage']['recordings_path']
        self.max_usage = int(self.config['storage']['max_usage_percent'])
        self.recording_duration = int(self.config['motion']['recording_duration'])
        self.threshold = int(self.config['motion']['threshold'])
        self.min_area = int(self.config['motion']['min_area'])

        # Telegram alert toggle (persisted in config)
        alerts_val = self.config.get('telegram', 'alerts_enabled', fallback='true')
        self.alerts_enabled = alerts_val.strip().lower() == 'true'

        # Motion zone: normalised floats 0.0–1.0 (x, y, w, h)
        # Falls back to full frame if not in config
        self.motion_zone = {
            'x': float(self.config.get('motion', 'zone_x', fallback='0.0')),
            'y': float(self.config.get('motion', 'zone_y', fallback='0.0')),
            'w': float(self.config.get('motion', 'zone_w', fallback='1.0')),
            'h': float(self.config.get('motion', 'zone_h', fallback='1.0')),
        }

        # Camera setup
        self.camera = cv2.VideoCapture(0, cv2.CAP_V4L2)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.config['camera']['width']))
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.config['camera']['height']))
        self.camera.set(cv2.CAP_PROP_FPS, int(self.config['camera']['fps']))
        
        self.frame_width  = int(self.config['camera']['width'])
        self.frame_height = int(self.config['camera']['height'])

        # Motion detection variables
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2()
        self.is_recording = False
        self.last_motion_time = 0
        
        os.makedirs(self.recordings_path, exist_ok=True)
        
        self.start_motion_detection()
        self.start_storage_monitor()
        self.send_startup_message()

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

    def send_telegram_message(self, message, override = False):
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
        message = f"🎥 Motion Detector Started\nHost: {host}\nWeb Interface: http://{ip}:5000"
        self.send_telegram_message(message, override=True)

    # ------------------------------------------------------------------
    # Camera / motion
    # ------------------------------------------------------------------

    def get_frame(self):
        ret, frame = self.camera.read()
        return frame if ret else None

    def _zone_roi(self, frame):
        """Crop frame to the current motion zone."""
        h, w = frame.shape[:2]
        zx = int(self.motion_zone['x'] * w)
        zy = int(self.motion_zone['y'] * h)
        zw = int(self.motion_zone['w'] * w)
        zh = int(self.motion_zone['h'] * h)
        # Clamp
        zx = max(0, min(zx, w - 1))
        zy = max(0, min(zy, h - 1))
        zw = max(1, min(zw, w - zx))
        zh = max(1, min(zh, h - zy))
        return frame[zy:zy+zh, zx:zx+zw], (zx, zy, zw, zh)

    def detect_motion(self, frame):
        roi, _ = self._zone_roi(frame)
        learning_rate = 0 if self.is_recording else 0.05
        fg_mask = self.background_subtractor.apply(roi, learningRate=learning_rate)

        # APPLY THRESHOLD HERE
        _, fg_mask = cv2.threshold(fg_mask, self.threshold, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion_detected = any(cv2.contourArea(c) > self.min_area for c in contours)

        return motion_detected, fg_mask

    def record_video(self):
        if self.is_recording:
            return
        self.is_recording = True
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.recordings_path, f"motion_{timestamp}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(self.config['camera']['fps'])
        width = int(self.config['camera']['width'])
        height = int(self.config['camera']['height'])
        out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
        start_time = time.time()
        print(f"Recording started: {filename}")
        try:
            while time.time() - start_time < self.recording_duration:
                t0 = time.time()
                frame = self.get_frame()
                if frame is not None:
                    # Burn timestamp into recorded file            <-- ADD THIS
                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    cv2.putText(frame, ts, (10, height - 40),
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
    # Streaming
    # ------------------------------------------------------------------

    def generate_frames(self):
        while True:
            frame = self.get_frame()
            if frame is None:
                time.sleep(0.1)
                continue

            # Motion detection + overlays (your existing code)
            motion_detected, _ = self.detect_motion(frame)

            # Draw motion zone rectangle
            fh, fw = frame.shape[:2]
            zx = int(self.motion_zone['x'] * fw)
            zy = int(self.motion_zone['y'] * fh)
            zw = int(self.motion_zone['w'] * fw)
            zh = int(self.motion_zone['h'] * fh)
            cv2.rectangle(frame, (zx, zy), (zx + zw, zy + zh), (255, 165, 0), 2)

            # Status overlay
            status = "RECORDING" if self.is_recording else "MONITORING"
            color = (0, 0, 255) if self.is_recording else (0, 255, 0)
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            if motion_detected:
                cv2.putText(frame, "MOTION DETECTED", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Alert status
            alert_text = "ALERTS: ON" if self.alerts_enabled else "ALERTS: OFF"
            alert_color = (0, 255, 100) if self.alerts_enabled else (0, 100, 255)
            cv2.putText(frame, alert_text, (10, fh - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, alert_color, 2)

            # Timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, timestamp, (10, fh - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Encode to JPEG - lower quality often helps Safari stability
            ret, buffer = cv2.imencode('.jpg', frame, [
                int(cv2.IMWRITE_JPEG_QUALITY), 65   # Try 70-80
            ])

            if not ret:
                time.sleep(0.05)
                continue

            frame_bytes = buffer.tobytes()

            # Strict formatting that Safari likes better
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n'
                b'Content-Length: ' + str(len(frame_bytes)).encode() + b'\r\n'
                b'\r\n' +          # Important blank line
                frame_bytes +
                b'\r\n')

            # Slight increase in delay helps Safari not choke
            time.sleep(0.05)   # ~20 fps – good balance

# Initialize
detector = MotionDetector()

# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(
        detector.generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame',
        headers={
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            'Pragma': 'no-cache',
            'Expires': '0'
        }
    )

@app.route('/status')
def status():
    return jsonify({
        'is_recording': detector.is_recording,
        'storage_usage': round(detector.get_storage_usage(), 2),
        'recordings_count': len([f for f in os.listdir(detector.recordings_path) if f.endswith('.mp4')]),
        'alerts_enabled': detector.alerts_enabled,
        'motion_zone': detector.motion_zone,
    })

@app.route('/recordings')
def recordings():
    recs = []
    for filename in os.listdir(detector.recordings_path):
        if filename.endswith('.mp4'):
            filepath = os.path.join(detector.recordings_path, filename)
            size = os.path.getsize(filepath)
            created = datetime.fromtimestamp(os.path.getctime(filepath))
            recs.append({
                'filename': filename,
                'size_mb': round(size / (1024*1024), 2),
                'created': created.strftime('%Y-%m-%d %H:%M:%S')
            })
    recs.sort(key=lambda x: x['created'], reverse=True)
    return jsonify(recs)

@app.route('/save_zone', methods=['POST'])
def save_zone():
    """
    Expects JSON: { "x": 0.1, "y": 0.1, "w": 0.8, "h": 0.8 }
    All values are normalised (0.0 – 1.0) relative to frame dimensions.
    """
    data = request.get_json()
    try:
        x = float(data['x'])
        y = float(data['y'])
        w = float(data['w'])
        h = float(data['h'])
        # Basic validation
        if not (0 <= x < 1 and 0 <= y < 1 and 0 < w <= 1 and 0 < h <= 1):
            raise ValueError("Values out of range")
        detector.save_motion_zone(x, y, w, h)
        return jsonify({'success': True, 'motion_zone': detector.motion_zone})
    except (KeyError, ValueError, TypeError) as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/toggle_alerts', methods=['POST'])
def toggle_alerts():
    """Toggle Telegram alerts on/off. Optionally pass { "enabled": true/false }."""
    data = request.get_json(silent=True) or {}
    if 'enabled' in data:
        new_state = bool(data['enabled'])
    else:
        new_state = not detector.alerts_enabled
    detector.save_alerts_enabled(new_state)
    return jsonify({'success': True, 'alerts_enabled': detector.alerts_enabled})

@app.route('/update_settings', methods=['POST'])
def update_settings():
    """
    Expects JSON with any combination of:
    { "threshold": 25, "min_area": 500, "recording_duration": 30 }
    """
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
            if not (5 <= val <= 600):
                raise ValueError("Must be between 5 and 600 seconds")
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


@app.route('/settings')
def get_settings():
    """Return current motion detection settings."""
    return jsonify({
        'threshold': detector.threshold,
        'min_area': detector.min_area,
        'recording_duration': detector.recording_duration,
    })

@app.route('/network_info')
def network_info():
    """Return hostname and local IP."""
    return jsonify({
        'hostname': detector.get_hostname(),
        'ip': detector.get_local_ip(),
    })


@app.route('/update_telegram', methods=['POST'])
def update_telegram():
    """
    Update Telegram credentials.
    Expects JSON: { "bot_token": "...", "chat_id": "..." }
    At least one field must be present.
    """
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
            updated['bot_token'] = True   # don't echo the token back

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

@app.route('/telegram')
def telegram_page():
    return render_template('telegram.html')

RECORDINGS_DIR = os.path.abspath("recordings")

def safe_path(filename):
    path = os.path.abspath(os.path.join(RECORDINGS_DIR, filename))
    if not path.startswith(RECORDINGS_DIR):
        raise ValueError("Invalid path")
    return path

@app.route('/recordings/<filename>')
def view_recording(filename):
    path = safe_path(filename)
    return send_from_directory(RECORDINGS_DIR, os.path.basename(path), mimetype='video/mp4', conditional=True, as_attachment=False)  # opens in browser

@app.route('/download/<filename>')
def download_recording(filename):
    try:
        path = safe_path(filename)
        return send_from_directory(RECORDINGS_DIR, os.path.basename(path), as_attachment=True)
    except:
        return "Invalid file", 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
