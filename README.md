Flask app runs on RPi 4B Trixie.
Tested on Windows and Mac host.
Browser works across all platform.

API
GET /api/recordings?page=1&per_page=20Paginated list; every item includes prev / next filenames and view_url / download_url
GET /api/recordings/<filename>/infoMetadata + neighbour links for one clip by filename
GET /api/recordings/<filename>/streamStreams the MP4 with HTTP Range support — enables seek/scrub from remote players
GET /api/recordings/latestShortcut returning info for the most recent clip
