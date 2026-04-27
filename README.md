# Motion Detector

A Python-based motion detection system that uses your webcam to detect movement, record video clips, and send alerts via Telegram. It includes a web interface for live monitoring and remote management of recordings.

---

## Features

- **Live video stream** with motion zone overlay and status indicators
- **Automatic recording** triggered by motion detection (with debounce to reduce false positives)
- **Telegram alerts** on motion detection, startup, and camera connect/disconnect events
- **Configurable motion zone** — draw a region of interest directly in the web UI
- **Storage management** — automatically deletes oldest recordings when disk usage exceeds a threshold
- **Camera watchdog** — detects disconnection and resumes automatically when the camera is reconnected
- **REST API** for remote access to recordings, with pagination, byte-range streaming, and neighbour navigation
- **Cross-platform** — supports Windows (DirectShow), macOS (AVFoundation), and Linux (V4L2)

---

## Requirements

- Python 3.8+
- A webcam accessible at device index `0`
- A Telegram bot token and chat ID (for alerts)

Install dependencies:

```bash
pip install opencv-python flask requests numpy
```

---

## Configuration

Copy `config_template.ini` to `config.ini` and fill in your values:

```ini
[telegram]
bot_token = YOUR_BOT_TOKEN
chat_id = YOUR_CHAT_ID
alerts_enabled = true

[storage]
recordings_path = recordings
max_usage_percent = 90

[camera]
width = 640
height = 480
fps = 20
rotation = 0        # 0, 90, 180, or 270

[motion]
recording_duration = 30     # seconds
threshold = 25              # binary threshold (1–255)
min_area = 500              # minimum contour area in pixels
zone_x = 0.0               # normalised 0.0–1.0
zone_y = 0.0
zone_w = 1.0
zone_h = 1.0
```

> **Note:** `config.ini` is required at startup. The application will raise a `FileNotFoundError` if it is missing.

---

## Usage

```bash
python app.py
```

The server will start on the first available port in the range `5000–5100`. A startup message with the host, IP address, and web interface URL will be sent to your Telegram chat.

Open the web interface in your browser:

```
http://<your-ip>:<port>
```

---

## Web Interface

| Page | URL | Description |
|---|---|---|
| Dashboard | `/` | Live feed, recording list, status |
| Telegram settings | `/telegram` | Update bot token, chat ID, test alerts |

---

## API Reference

All API endpoints return JSON unless otherwise noted.

### Status & Settings

| Method | Endpoint | Description |
|---|---|---|
| GET | `/status` | Camera state, recording status, storage usage, alert toggle |
| GET | `/settings` | Current motion detection settings |
| GET | `/network_info` | Hostname and local IP |
| POST | `/update_settings` | Update `threshold`, `min_area`, `recording_duration` |
| POST | `/set_rotation` | Set camera rotation (`0`, `90`, `180`, `270`) |
| POST | `/save_zone` | Set motion zone (`x`, `y`, `w`, `h` as normalised floats) |
| POST | `/toggle_alerts` | Enable or disable Telegram alerts |
| POST | `/update_telegram` | Update `bot_token` and/or `chat_id` |
| POST | `/test_telegram` | Send a test Telegram message |

### Live Feed

| Method | Endpoint | Description |
|---|---|---|
| GET | `/video_feed` | MJPEG stream |
| GET | `/snapshot` | Single JPEG frame |

### Recordings

| Method | Endpoint | Description |
|---|---|---|
| GET | `/recordings` | Full list, newest first, with index |
| GET | `/recordings/<filename>` | Serve recording for in-browser playback |
| GET | `/download/<filename>` | Download recording as attachment |
| GET | `/api/recordings` | Paginated list (`?page=1&per_page=20`) |
| GET | `/api/recordings/latest` | Metadata for the most recent recording |
| GET | `/api/recordings/<filename>/info` | Metadata + prev/next neighbours for one recording |
| GET | `/api/recordings/<filename>/stream` | Byte-range streaming (supports seek/scrub) |

---

## Project Structure

```
.
├── app.py
├── config.ini              # Your local config (not committed)
├── config_template.ini     # Template to copy from
├── recordings/             # Auto-created; stores .mp4 files
└── templates/
    ├── index.html
    └── telegram.html
```

---

## Notes

- Recordings are saved as `.mp4` files using the `mp4v` codec. Some browsers may require re-encoding for direct playback; the byte-range streaming endpoint is provided to improve compatibility with remote players.
- The background subtractor learning rate is frozen during recording and for 3 seconds afterwards to prevent the end of a recording from immediately triggering another.
- Motion must be detected in at least 3 consecutive frames before a recording is triggered, reducing false positives from noise or lighting changes.
- All settings changes made via the API or web UI are persisted back to `config.ini` immediately.
