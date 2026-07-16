"""
run_with_ngrok.py
==================
Launches the LEONI MED Streamlit app (app.py) locally and exposes it to the
public internet through an ngrok tunnel.

Usage:
    python run_with_ngrok.py

Requirements:
    pip install streamlit pyngrok
    (plus tensorflow, pillow, scikit-learn, seaborn, matplotlib for app.py itself)
"""

import atexit
import subprocess
import sys
import time

from pyngrok import ngrok, conf

# ------------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------------
NGROK_AUTH_TOKEN = "3Cd8WmMhgUVg6mnMTqjS6bP03ja_5Q7R66ifg6GaNKPKgHYJc"
STREAMLIT_PORT = 8501
APP_FILE = "app.py"

# ------------------------------------------------------------------------------------
# Configure ngrok
# ------------------------------------------------------------------------------------
conf.get_default().auth_token = NGROK_AUTH_TOKEN
ngrok.set_auth_token(NGROK_AUTH_TOKEN)

# ------------------------------------------------------------------------------------
# Start the Streamlit app as a subprocess
# ------------------------------------------------------------------------------------
print(f"Starting Streamlit app '{APP_FILE}' on port {STREAMLIT_PORT} ...")

streamlit_process = subprocess.Popen(
    [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        APP_FILE,
        "--server.port",
        str(STREAMLIT_PORT),
        "--server.headless",
        "true",
        "--server.address",
        "0.0.0.0",
    ]
)


def cleanup():
    print("\nShutting down Streamlit and ngrok tunnel...")
    ngrok.kill()
    streamlit_process.terminate()


atexit.register(cleanup)

# Give Streamlit a moment to boot before opening the tunnel
time.sleep(5)

# ------------------------------------------------------------------------------------
# Open the ngrok tunnel
# ------------------------------------------------------------------------------------
public_tunnel = ngrok.connect(STREAMLIT_PORT, "http")
print("=" * 70)
print(f"  LEONI MED app is live at: {public_tunnel.public_url}")
print("=" * 70)
print("Press Ctrl+C to stop the app and close the tunnel.")

# ------------------------------------------------------------------------------------
# Keep the script alive until interrupted
# ------------------------------------------------------------------------------------
try:
    streamlit_process.wait()
except KeyboardInterrupt:
    pass
