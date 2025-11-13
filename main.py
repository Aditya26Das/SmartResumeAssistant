import subprocess
import sys
import os
import webbrowser
import socket
from dotenv import load_dotenv

def find_free_port(default_port=8501):
    """Find a free port if the default one is in use."""
    port = default_port
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("localhost", port))
                return port
            except OSError:
                port += 1

def run_streamlit_app():
    """
    Automatically runs the Smart ATS Streamlit app.
    Handles .env, browser launch, and port conflicts automatically.
    """
    app_file = "app.py"
    load_dotenv()
    try:
        import streamlit
    except ImportError:
        print("⚠️ Streamlit not found. Installing it now...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
        import streamlit
    port = find_free_port(8501)
    os.environ["STREAMLIT_SERVER_PORT"] = str(port)
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"

    url = f"http://localhost:{port}"
    print(f"\n🚀 Launching Smart ATS Streamlit App on {url}")
    webbrowser.open(url, new=2)

    result = subprocess.run(
        [sys.executable, "-m", "streamlit", "run", app_file],
        stdout=sys.stdout,
        stderr=sys.stderr
    )

    if result.returncode == 0:
        print("✅ Streamlit app closed successfully.")
    else:
        print(f"❌ Streamlit app exited with code {result.returncode}")

if __name__ == "__main__":
    run_streamlit_app()
