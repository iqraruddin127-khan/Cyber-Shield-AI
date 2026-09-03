import socket
import subprocess
import sys
import threading
import time

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BACKEND_HOST = "127.0.0.1"
BACKEND_PORT = 8000
FRONTEND_PORT = 8501
SUPABASE_REDIRECT_URL = f"http://{BACKEND_HOST}:{FRONTEND_PORT}" 
HEALTH_URL = f"http://{BACKEND_HOST}:{BACKEND_PORT}/api/health"
HEALTH_POLL_INTERVAL = 0.5   # seconds between health-check attempts
HEALTH_POLL_TIMEOUT = 15.0   # total seconds to wait for backend readiness


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """Return True if a TCP listener is already bound to the given port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex((host, port)) == 0


def wait_for_backend() -> bool:
    """
    Poll the backend health endpoint until it responds or the timeout
    elapses.  Returns True if the backend became ready, False otherwise.
    """
    deadline = time.monotonic() + HEALTH_POLL_TIMEOUT
    while time.monotonic() < deadline:
        try:
            response = requests.get(HEALTH_URL, timeout=2)
            if response.status_code == 200:
                return True
        except requests.exceptions.ConnectionError:
            pass
        except requests.exceptions.Timeout:
            pass
        time.sleep(HEALTH_POLL_INTERVAL)
    return False


def kill_process(proc: subprocess.Popen, label: str) -> None:
    """Terminate a child process, tolerating already-dead processes."""
    if proc.poll() is None:  # still running
        try:
            proc.kill()
            print(f"  ✓ {label} process terminated.")
        except OSError as exc:
            print(f"  ⚠ Could not kill {label} process: {exc}")


def monitor_process(proc: subprocess.Popen, label: str, peer: subprocess.Popen, peer_label: str) -> None:
    """
    Background watcher that prints a warning if the monitored process exits
    unexpectedly and, in response, tears down its peer.
    """
    proc.wait()
    # If we get here the process has exited
    if proc.returncode != 0:
        print(f"\n⚠️  {label} exited unexpectedly (code {proc.returncode}).")
        print(f"    Shutting down {peer_label}…")
        kill_process(peer, peer_label)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def start_cyber_shield() -> None:
    print("🛡️  Starting Cyber Shield AI Core Ecosystem…")

    # ---- Pre-flight port checks ----
    for port, service in [(BACKEND_PORT, "Backend (FastAPI)"), (FRONTEND_PORT, "Frontend (Streamlit)")]:
        if is_port_in_use(port):
            print(f"❌ Port {port} is already in use — cannot start {service}.")
            print("   Please free the port and try again.")
            sys.exit(1)

    # ---- 1. Start FastAPI backend ----
    print("📡 Launching AI Backend Server…")
    backend_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app:app",
         "--host", BACKEND_HOST, "--port", str(BACKEND_PORT)],
    )

    # ---- 2. Wait for backend readiness ----
    print(f"   Waiting for backend to become ready (up to {HEALTH_POLL_TIMEOUT:.0f}s)…")
    if not wait_for_backend():
        print("❌ Backend did not become ready in time. Aborting.")
        kill_process(backend_process, "Backend")
        sys.exit(1)
    print("   ✓ Backend is ready.")

    # ---- 3. Start Streamlit frontend ----
    print("🎨 Launching Antivirus Command Center UI…")
    frontend_process = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "dashboard.py"],
    )

    # ---- 4. Start background monitors ----
    t_backend = threading.Thread(
        target=monitor_process,
        args=(backend_process, "Backend", frontend_process, "Frontend"),
        daemon=True,
    )
    t_frontend = threading.Thread(
        target=monitor_process,
        args=(frontend_process, "Frontend", backend_process, "Backend"),
        daemon=True,
    )
    t_backend.start()
    t_frontend.start()

    # ---- 5. Keep main script alive until both children exit ----
    try:
        while backend_process.poll() is None or frontend_process.poll() is None:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down Cyber Shield AI Ecosystem cleanly…")
        kill_process(backend_process, "Backend")
        kill_process(frontend_process, "Frontend")
        print("🛑 Shutdown complete.")


if __name__ == "__main__":
    start_cyber_shield()
