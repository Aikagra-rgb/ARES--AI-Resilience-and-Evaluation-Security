#!/usr/bin/env python3
"""
ARES Launching Terminal Engine.

Unified process launcher for the ARES platform:
- Frontend (.NET 8.0 Blazor Server in src/Ares.Web on http://localhost:5149)
- Backend (FastAPI in backend on http://127.0.0.1:8000)
- Both (concurrent execution with prefixed logs, readiness probe, clean Ctrl+C shutdown)
- LoRA (PEFT inference server on http://localhost:11435)
- All (Frontend + Backend + LoRA full stack)
"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

# Default configurations
DEFAULT_BACKEND_PORT = 8000
DEFAULT_FRONTEND_PORT = 5149
DEFAULT_LORA_PORT = 11435
DEFAULT_HOST = "127.0.0.1"
HEALTH_CHECK_TIMEOUT = 45.0
HEALTH_CHECK_INTERVAL = 0.5

# ANSI Color codes
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_BLUE = "\033[94m"
    BG_BLUE = "\033[44m"


_COLORS_ENABLED = True


def set_colors_enabled(enabled: bool) -> None:
    global _COLORS_ENABLED
    _COLORS_ENABLED = enabled


def colorize(text: str, color: str) -> str:
    if not _COLORS_ENABLED:
        return text
    return f"{color}{text}{Colors.RESET}"


def init_windows_vt_mode() -> None:
    """Enable VT100 escape codes on Windows console if available."""
    if sys.platform == "win32":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
            mode = ctypes.c_ulong()
            if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                # ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
                kernel32.SetConsoleMode(handle, mode.value | 0x0004)
        except Exception:
            pass


_LOG_LOCK = threading.Lock()


def log_message(tag: str, message: str, color: str = Colors.WHITE) -> None:
    timestamp = time.strftime("%H:%M:%S")
    formatted_tag = f"[{tag:<8}]"
    with _LOG_LOCK:
        print(f"{colorize(timestamp, Colors.DIM)} {colorize(formatted_tag, color)} {message}", flush=True)


@dataclass
class EnvironmentInfo:
    repo_root: Path
    backend_dir: Path
    frontend_dir: Path
    python_path: Path
    dotnet_path: Path | None
    venv_path: Path | None
    is_windows: bool
    dotnet_version: str | None = None
    python_version: str | None = None


def detect_environment(repo_root: Path | None = None) -> EnvironmentInfo:
    """Detect available Python, .NET SDK, and project directories."""
    if repo_root is None:
        repo_root = Path(__file__).resolve().parent

    backend_dir = repo_root / "backend"
    frontend_dir = repo_root / "src" / "Ares.Web"
    is_windows = sys.platform == "win32"

    # 1. Detect Python executable
    python_candidates = [
        repo_root / "venv" / ("Scripts" if is_windows else "bin") / ("python.exe" if is_windows else "python"),
        repo_root / ".venv" / ("Scripts" if is_windows else "bin") / ("python.exe" if is_windows else "python"),
        backend_dir / "venv" / ("Scripts" if is_windows else "bin") / ("python.exe" if is_windows else "python"),
        backend_dir / ".venv" / ("Scripts" if is_windows else "bin") / ("python.exe" if is_windows else "python"),
    ]

    selected_python: Path | None = None
    detected_venv: Path | None = None

    for candidate in python_candidates:
        if candidate.is_file():
            # Check if this candidate can execute cleanly and load standard C extensions
            is_valid = False
            try:
                res = subprocess.run(
                    [str(candidate), "-c", "import sys, ssl; print(sys.version.split()[0])"],
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
                if res.returncode == 0:
                    selected_python = candidate
                    detected_venv = candidate.parent.parent
                    is_valid = True
                    break
            except Exception:
                pass

            # If Windows WDAC or DLL issues exist, inspect pyvenv.cfg for base executable
            if not is_valid:
                pyvenv_cfg = candidate.parent.parent / "pyvenv.cfg"
                if pyvenv_cfg.is_file():
                    try:
                        content = pyvenv_cfg.read_text(encoding="utf-8")
                        for line in content.splitlines():
                            if line.startswith("executable ="):
                                base_exe = Path(line.split("=", 1)[1].strip())
                                if base_exe.is_file():
                                    selected_python = base_exe
                                    detected_venv = candidate.parent.parent
                                    break
                        if selected_python:
                            break
                    except Exception:
                        pass

    if selected_python is None:
        selected_python = Path(sys.executable)

    # 2. Detect .NET CLI
    dotnet_candidate = shutil.which("dotnet")
    dotnet_path = Path(dotnet_candidate) if dotnet_candidate else None
    if dotnet_path is None and is_windows:
        default_win_dotnet = Path("C:/Program Files/dotnet/dotnet.exe")
        if default_win_dotnet.is_file():
            dotnet_path = default_win_dotnet

    # 3. Retrieve versions
    python_ver = None
    try:
        res = subprocess.run([str(selected_python), "--version"], capture_output=True, text=True, timeout=3)
        if res.returncode == 0:
            python_ver = res.stdout.strip() or res.stderr.strip()
    except Exception:
        python_ver = f"Python {sys.version.split()[0]}"

    dotnet_ver = None
    if dotnet_path and dotnet_path.is_file():
        try:
            res = subprocess.run([str(dotnet_path), "--version"], capture_output=True, text=True, timeout=5)
            if res.returncode == 0 and res.stdout.strip():
                dotnet_ver = res.stdout.strip()
            else:
                dotnet_ver = ".NET 8.0 SDK (detected)"
        except Exception:
            dotnet_ver = ".NET 8.0 SDK (detected)"

    return EnvironmentInfo(
        repo_root=repo_root,
        backend_dir=backend_dir,
        frontend_dir=frontend_dir,
        python_path=selected_python,
        dotnet_path=dotnet_path,
        venv_path=detected_venv,
        is_windows=is_windows,
        dotnet_version=dotnet_ver,
        python_version=python_ver,
    )


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """Check if a network port is actively in use (IPv4 or IPv6)."""
    if get_pids_for_port(port):
        return True

    # First attempt: connect to port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        try:
            s.connect((host, port))
            return True
        except (TimeoutError, ConnectionRefusedError, OSError):
            pass

    # Second attempt: try to bind port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
            return False
        except OSError:
            return True


def get_pids_for_port(port: int) -> list[int]:
    """Find process IDs currently listening on a specific port."""
    pids: list[int] = []
    if sys.platform == "win32":
        try:
            res = subprocess.run(
                ["netstat", "-ano", "-p", "tcp"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if res.returncode == 0:
                for line in res.stdout.splitlines():
                    tokens = line.strip().split()
                    # Expect: Proto  Local_Address  Foreign_Address  State  PID
                    if len(tokens) >= 5 and tokens[0].upper() == "TCP":
                        local_addr = tokens[1]
                        state = tokens[3].upper()
                        pid_str = tokens[4]
                        # Match :port at end of local address (e.g., 0.0.0.0:8000, 127.0.0.1:8000, [::]:8000)
                        if (local_addr.endswith(f":{port}") or local_addr.endswith(f"]:{port}")) and state == "LISTENING" and pid_str.isdigit():
                            pids.append(int(pid_str))
        except Exception:
            pass
    else:
        try:
            res = subprocess.run(
                ["lsof", "-ti", f":{port}"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if res.returncode == 0:
                for line in res.stdout.splitlines():
                    val = line.strip()
                    if val.isdigit():
                        pids.append(int(val))
        except Exception:
            pass

    return sorted(set(pids))


def get_child_pids(parent_pid: int) -> list[int]:
    """Find all direct child process IDs for a given parent PID."""
    if parent_pid <= 0:
        return []
    pids: list[int] = []
    if sys.platform == "win32":
        try:
            res = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    f"Get-CimInstance Win32_Process -Filter 'ParentProcessId = {parent_pid}' | Select-Object -ExpandProperty ProcessId",
                ],
                capture_output=True,
                text=True,
                timeout=3,
            )
            if res.returncode == 0:
                for line in res.stdout.splitlines():
                    val = line.strip()
                    if val.isdigit():
                        pids.append(int(val))
        except Exception:
            pass
    else:
        try:
            res = subprocess.run(
                ["pgrep", "-P", str(parent_pid)],
                capture_output=True,
                text=True,
                timeout=3,
            )
            if res.returncode == 0:
                for line in res.stdout.splitlines():
                    val = line.strip()
                    if val.isdigit():
                        pids.append(int(val))
        except Exception:
            pass
    return sorted(set(pids))


def kill_process_tree(pid: int) -> bool:
    """Forcefully terminate a process and all of its child processes."""
    if pid <= 0:
        return False
    if sys.platform == "win32":
        try:
            res = subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return res.returncode == 0
        except Exception:
            return False
    else:
        try:
            pgid = os.getpgid(pid)
            if pgid != os.getpgrp():
                os.killpg(pgid, signal.SIGKILL)
            else:
                os.kill(pid, signal.SIGKILL)
            return True
        except Exception:
            try:
                os.kill(pid, signal.SIGKILL)
                return True
            except Exception:
                return False


def clean_ports(ports: list[int], verbose: bool = True) -> int:
    """Terminate all processes currently bound to specified ports."""
    total_killed = 0
    for port in ports:
        pids = get_pids_for_port(port)
        for pid in pids:
            if verbose:
                log_message("CLEAN", f"Terminating process {pid} holding port {port}...", Colors.YELLOW)
            if kill_process_tree(pid):
                total_killed += 1

    # Verify ports freed
    time.sleep(0.5)
    remaining: list[int] = []
    for port in ports:
        if is_port_in_use(port):
            remaining.append(port)

    if remaining and verbose:
        log_message("WARN", f"Ports still occupied after cleanup: {remaining}", Colors.YELLOW)
    elif verbose and total_killed > 0:
        log_message("CLEAN", f"Successfully freed all target ports ({ports}).", Colors.GREEN)
    elif verbose:
        log_message("CLEAN", f"All target ports ({ports}) are clean and ready.", Colors.GREEN)

    return total_killed


def build_child_env(env_info: EnvironmentInfo) -> dict[str, str]:
    """Construct environment variables for spawned child processes."""
    child_env = os.environ.copy()
    child_env["PYTHONUNBUFFERED"] = "1"

    # Configure PYTHONPATH to prioritize backend and site-packages
    pythonpaths = [str(env_info.backend_dir)]
    if env_info.venv_path:
        site_packages = (
            env_info.venv_path / "Lib" / "site-packages"
            if env_info.is_windows
            else env_info.venv_path / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
        )
        if site_packages.is_dir():
            pythonpaths.append(str(site_packages))

    existing_pythonpath = child_env.get("PYTHONPATH", "")
    if existing_pythonpath:
        pythonpaths.append(existing_pythonpath)

    child_env["PYTHONPATH"] = os.pathsep.join(pythonpaths)
    if env_info.venv_path:
        child_env["VIRTUAL_ENV"] = str(env_info.venv_path)

    return child_env


@dataclass
class ServiceTask:
    name: str
    tag: str
    color: str
    command: list[str]
    cwd: Path
    env: dict[str, str]
    port: int
    health_url: str | None = None


class ProcessManager:
    """Manages concurrent child processes, log streaming, and graceful shutdown."""

    def __init__(self) -> None:
        self.processes: list[tuple[ServiceTask, subprocess.Popen]] = []
        self._threads: list[threading.Thread] = []
        self._stopping = threading.Event()
        self._stopped = False
        self._lock = threading.Lock()

    def start_service(self, task: ServiceTask) -> subprocess.Popen:
        """Start a service in a background child process with streaming logs."""
        creationflags = 0
        preexec_fn = None
        if sys.platform == "win32":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            preexec_fn = os.setsid

        log_message(task.tag, f"Launching {task.name}: {' '.join(task.command)}", task.color)

        proc = subprocess.Popen(
            task.command,
            cwd=str(task.cwd),
            env=task.env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            creationflags=creationflags,
            preexec_fn=preexec_fn,
        )

        with self._lock:
            self.processes.append((task, proc))

        # Start streaming thread
        t = threading.Thread(
            target=self._stream_output,
            args=(task, proc),
            daemon=True,
            name=f"stream-{task.name}",
        )
        t.start()
        self._threads.append(t)
        return proc

    def _stream_output(self, task: ServiceTask, proc: subprocess.Popen) -> None:
        """Stream process stdout/stderr line-by-line with prefixed colored tags."""
        try:
            assert proc.stdout is not None
            for line in iter(proc.stdout.readline, ""):
                if self._stopping.is_set() and not line:
                    break
                stripped = line.rstrip("\r\n")
                if stripped:
                    log_message(task.tag, stripped, task.color)
        except Exception:
            pass

    def stop_all(self, ports: list[int] | None = None) -> None:
        """Gracefully stop all child processes without leaving orphans."""
        with self._lock:
            if self._stopped:
                return
            self._stopped = True
            self._stopping.set()
            active_procs = list(self.processes)

        log_message("ARES", "Initiating shutdown of all running processes...", Colors.BRIGHT_YELLOW)

        # Pre-collect all descendant child PIDs while root processes are still alive
        pids_to_kill: set[int] = set()
        for _, proc in active_procs:
            if proc.pid:
                pids_to_kill.add(proc.pid)
                for child_pid in get_child_pids(proc.pid):
                    pids_to_kill.add(child_pid)

        # Step 1: Send graceful termination signal to process trees
        for task, proc in active_procs:
            if proc.poll() is None:
                log_message(task.tag, f"Stopping {task.name} (PID: {proc.pid})...", task.color)
                if sys.platform == "win32":
                    try:
                        # Process was spawned with CREATE_NEW_PROCESS_GROUP
                        proc.send_signal(signal.CTRL_BREAK_EVENT)
                    except Exception:
                        try:
                            # Attempt graceful taskkill without /F first
                            subprocess.run(["taskkill", "/T", "/PID", str(proc.pid)], capture_output=True, timeout=2)
                        except Exception:
                            pass
                else:
                    try:
                        pgid = os.getpgid(proc.pid)
                        if pgid != os.getpgrp():
                            os.killpg(pgid, signal.SIGTERM)
                        else:
                            os.kill(proc.pid, signal.SIGTERM)
                    except Exception:
                        pass

        # Step 2: Wait briefly for clean termination
        time.sleep(1.0)

        # Step 3: Forcefully kill all collected process trees and child PIDs
        for pid in pids_to_kill:
            kill_process_tree(pid)

        # Step 4: Ensure target ports are completely free
        if ports:
            clean_ports(ports, verbose=False)

        log_message("ARES", "All processes terminated cleanly. Ports released.", Colors.BRIGHT_GREEN)

    def wait(self) -> int:
        """Wait for any child process to exit or monitor until interrupt."""
        try:
            while not self._stopping.is_set():
                with self._lock:
                    for task, proc in self.processes:
                        code = proc.poll()
                        if code is not None:
                            log_message(
                                task.tag,
                                f"Process {task.name} (PID: {proc.pid}) exited with code {code}",
                                Colors.YELLOW if code == 0 else Colors.RED,
                            )
                            return code
                time.sleep(0.5)
        except KeyboardInterrupt:
            return 0
        return 0


def probe_backend_health(
    health_url: str,
    backend_port: int,
    frontend_port: int | None = None,
    lora_port: int | None = None,
    timeout: float = HEALTH_CHECK_TIMEOUT,
    interval: float = HEALTH_CHECK_INTERVAL,
    stop_event: threading.Event | None = None,
) -> bool:
    """Poll backend health endpoint until ready and display service status card."""
    log_message("HEALTH", f"Probing backend readiness at {health_url}...", Colors.CYAN)
    start_time = time.time()

    while time.time() - start_time < timeout:
        if stop_event and stop_event.is_set():
            return False

        try:
            req = urllib.request.Request(
                health_url,
                headers={"User-Agent": "ARES-Terminal-Launcher/1.0", "Accept": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                if resp.status in (200, 201):
                    elapsed = round(time.time() - start_time, 2)
                    log_message("HEALTH", f"Backend responded in {elapsed}s (HTTP {resp.status})", Colors.BRIGHT_GREEN)
                    _print_service_card(backend_port, frontend_port, lora_port)
                    return True
        except (TimeoutError, urllib.error.URLError, ConnectionError, OSError):
            pass

        time.sleep(interval)

    log_message(
        "WARN",
        f"Backend health check timed out after {timeout}s (service may still be initializing)",
        Colors.YELLOW,
    )
    return False


def _print_service_card(backend_port: int, frontend_port: int | None, lora_port: int | None) -> None:
    """Print formatted terminal card with clickable service URLs."""
    divider = "=" * 64
    lines = [
        "",
        colorize(divider, Colors.BRIGHT_CYAN),
        colorize("  * ARES Platform Services are Active & Ready *", Colors.BOLD + Colors.BRIGHT_GREEN),
        colorize(divider, Colors.BRIGHT_CYAN),
        f"  * Backend API:        {colorize(f'http://127.0.0.1:{backend_port}', Colors.BRIGHT_CYAN)}",
        f"  * Interactive Docs:   {colorize(f'http://127.0.0.1:{backend_port}/docs', Colors.BRIGHT_CYAN)}",
        f"  * Health Probe:       {colorize(f'http://127.0.0.1:{backend_port}/health', Colors.BRIGHT_CYAN)}",
    ]

    if frontend_port:
        lines.append(
            f"  * Web Console:        {colorize(f'http://localhost:{frontend_port}', Colors.BRIGHT_MAGENTA)}"
        )

    if lora_port:
        lines.append(
            f"  * LoRA Inference:     {colorize(f'http://localhost:{lora_port}', Colors.BRIGHT_YELLOW)}"
        )

    lines.extend([
        colorize("  * Press Ctrl+C at any time to shut down all services cleanly", Colors.DIM),
        colorize(divider, Colors.BRIGHT_CYAN),
        "",
    ])

    with _LOG_LOCK:
        print("\n".join(lines), flush=True)


def display_ascii_banner(env_info: EnvironmentInfo) -> None:
    """Display ARES header and detected environment diagnostics."""
    banner = r"""
    ___    ____  ______ _____ 
   /   |  / __ \/ ____// ___/ 
  / /| | / /_/ / __/   \__ \  
 / ___ |/ _, _/ /___  ___/ /  
/_/  |_/_/ |_/_____/ /____/   
 AI Resilience & Evaluation Security
    """
    print(colorize(banner, Colors.BRIGHT_CYAN))
    print(colorize("-" * 64, Colors.DIM))
    print(f" {colorize('Root:', Colors.BOLD)}    {env_info.repo_root}")
    print(f" {colorize('Python:', Colors.BOLD)}  {env_info.python_version} ({env_info.python_path})")
    dotnet_status = env_info.dotnet_version if env_info.dotnet_version else colorize("Not Found", Colors.YELLOW)
    print(f" {colorize('.NET:', Colors.BOLD)}    {dotnet_status}")
    print(colorize("-" * 64, Colors.DIM))


def check_ports_status(ports: list[tuple[str, int]]) -> None:
    """Display tabular status of target ports."""
    print("\n" + colorize("ARES Port & Service Status:", Colors.BOLD))
    print("+---------------+-------+--------------+------------------+")
    print("| Service       | Port  | Status       | Active PID(s)    |")
    print("+---------------+-------+--------------+------------------+")
    for name, port in ports:
        in_use = is_port_in_use(port)
        pids = get_pids_for_port(port)
        status_str = colorize("IN USE", Colors.YELLOW) if in_use else colorize("FREE", Colors.GREEN)
        pids_str = ", ".join(map(str, pids)) if pids else "-"
        print(f"| {name:<13} | {port:<5} | {status_str:<21} | {pids_str:<16} |")
    print("+---------------+-------+--------------+------------------+\n")


def show_interactive_menu(env_info: EnvironmentInfo) -> str:
    """Display terminal menu for interactive target selection."""
    display_ascii_banner(env_info)
    check_ports_status([
        ("Backend", DEFAULT_BACKEND_PORT),
        ("Frontend", DEFAULT_FRONTEND_PORT),
        ("LoRA Server", DEFAULT_LORA_PORT),
    ])

    print(colorize("Select an action to launch:", Colors.BOLD))
    print(f"  {colorize('[1]', Colors.BRIGHT_GREEN)} Both        - Launch Frontend & Backend concurrently {colorize('(Recommended)', Colors.DIM)}")
    print(f"  {colorize('[2]', Colors.BRIGHT_BLUE)} Backend     - Launch FastAPI Backend only (:8000)")
    print(f"  {colorize('[3]', Colors.BRIGHT_MAGENTA)} Frontend    - Launch Blazor Server Frontend only (:5149)")
    print(f"  {colorize('[4]', Colors.BRIGHT_CYAN)} All         - Launch Full Stack (Frontend + Backend + LoRA)")
    print(f"  {colorize('[5]', Colors.BRIGHT_YELLOW)} LoRA        - Launch LoRA Inference Server only (:11435)")
    print(f"  {colorize('[6]', Colors.WHITE)} Status      - Refresh port & service diagnostics")
    print(f"  {colorize('[7]', Colors.YELLOW)} Clean       - Free all ARES ports (kill orphaned processes)")
    print(f"  {colorize('[8]', Colors.RED)} Exit")
    print()

    choices = {
        "1": "both",
        "2": "backend",
        "3": "frontend",
        "4": "all",
        "5": "lora",
        "6": "status",
        "7": "clean",
        "8": "exit",
    }

    while True:
        try:
            raw = input(colorize("Enter selection [1-8]: ", Colors.BOLD)).strip()
            if raw in choices:
                return choices[raw]
            if raw.lower() in ("both", "backend", "frontend", "all", "lora", "status", "clean", "exit"):
                return raw.lower()
            print(colorize("Invalid option. Please choose between 1 and 8.", Colors.RED))
        except (KeyboardInterrupt, EOFError):
            print()
            return "exit"


def build_service_tasks(
    target: str,
    env_info: EnvironmentInfo,
    backend_port: int = DEFAULT_BACKEND_PORT,
    frontend_port: int = DEFAULT_FRONTEND_PORT,
    lora_port: int = DEFAULT_LORA_PORT,
    host: str = DEFAULT_HOST,
    reload: bool = True,
) -> list[ServiceTask]:
    """Construct execution task specifications for requested target services."""
    child_env = build_child_env(env_info)
    tasks: list[ServiceTask] = []

    # Backend task
    if target in ("backend", "both", "all"):
        backend_cmd = [
            str(env_info.python_path),
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            host,
            "--port",
            str(backend_port),
        ]
        if reload:
            backend_cmd.append("--reload")

        tasks.append(
            ServiceTask(
                name="FastAPI Backend",
                tag="BACKEND",
                color=Colors.BRIGHT_BLUE,
                command=backend_cmd,
                cwd=env_info.backend_dir,
                env=child_env,
                port=backend_port,
                health_url=f"http://127.0.0.1:{backend_port}/health",
            )
        )

    # Frontend task
    if target in ("frontend", "both", "all"):
        if not env_info.dotnet_path:
            raise RuntimeError(
                ".NET SDK CLI ('dotnet') was not found in PATH. "
                "Please install .NET 8.0 SDK from https://dotnet.microsoft.com/download/dotnet/8.0 "
                "to run the Ares.Web Blazor frontend."
            )

        frontend_env = child_env.copy()
        frontend_env["Ares__ApiBaseUrl"] = f"http://127.0.0.1:{backend_port}"
        frontend_env["ASPNETCORE_URLS"] = f"http://localhost:{frontend_port};http://127.0.0.1:{frontend_port}"

        frontend_cmd = [
            str(env_info.dotnet_path),
            "run",
            "--project",
            str(env_info.frontend_dir),
            "--urls",
            f"http://localhost:{frontend_port}",
        ]

        tasks.append(
            ServiceTask(
                name="Blazor Frontend",
                tag="FRONTEND",
                color=Colors.BRIGHT_MAGENTA,
                command=frontend_cmd,
                cwd=env_info.repo_root,
                env=frontend_env,
                port=frontend_port,
                health_url=f"http://localhost:{frontend_port}",
            )
        )

    # LoRA task
    if target in ("lora", "all"):
        lora_cmd = [
            str(env_info.python_path),
            "-m",
            "ares.training.serve_lora",
            "--port",
            str(lora_port),
        ]

        tasks.append(
            ServiceTask(
                name="LoRA Server",
                tag="LORA",
                color=Colors.BRIGHT_YELLOW,
                command=lora_cmd,
                cwd=env_info.backend_dir,
                env=child_env,
                port=lora_port,
                health_url=f"http://127.0.0.1:{lora_port}/healthz",
            )
        )

    return tasks


def launch_engine(
    target: str,
    env_info: EnvironmentInfo,
    backend_port: int = DEFAULT_BACKEND_PORT,
    frontend_port: int = DEFAULT_FRONTEND_PORT,
    lora_port: int = DEFAULT_LORA_PORT,
    host: str = DEFAULT_HOST,
    reload: bool = True,
    skip_health_check: bool = False,
    auto_clean_ports: bool = True,
    health_timeout: float = HEALTH_CHECK_TIMEOUT,
) -> int:
    """Main launch engine coordinator: verifies ports, spawns services, runs health probe, and monitors."""
    target_ports: list[int] = []
    if target in ("backend", "both", "all"):
        target_ports.append(backend_port)
    if target in ("frontend", "both", "all"):
        target_ports.append(frontend_port)
    if target in ("lora", "all"):
        target_ports.append(lora_port)

    # Port availability and automatic cleanup
    occupied_ports = [p for p in target_ports if is_port_in_use(p)]
    if occupied_ports:
        if auto_clean_ports:
            log_message("CLEAN", f"Cleaning existing processes on port(s): {occupied_ports}...", Colors.YELLOW)
            clean_ports(occupied_ports)
            still_occupied = [p for p in target_ports if is_port_in_use(p)]
            if still_occupied:
                log_message(
                    "ERROR",
                    f"Port(s) {still_occupied} could not be freed after cleanup. "
                    "Aborting launch to avoid port collision.",
                    Colors.RED,
                )
                return 1
        else:
            log_message(
                "ERROR",
                f"Required port(s) already in use: {occupied_ports}. Use --clean or free the ports first.",
                Colors.RED,
            )
            return 1

    tasks = build_service_tasks(
        target=target,
        env_info=env_info,
        backend_port=backend_port,
        frontend_port=frontend_port,
        lora_port=lora_port,
        host=host,
        reload=reload,
    )

    manager = ProcessManager()

    # Register OS signal handlers for clean teardown
    def handle_signal(sig, frame):
        log_message("ARES", f"Received signal {sig}. Initiating shutdown...", Colors.YELLOW)
        manager.stop_all(ports=target_ports)
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # Launch services
    for task in tasks:
        manager.start_service(task)

    # Health checking probe
    if not skip_health_check and target in ("backend", "both", "all"):
        health_thread = threading.Thread(
            target=probe_backend_health,
            kwargs={
                "health_url": f"http://127.0.0.1:{backend_port}/health",
                "backend_port": backend_port,
                "frontend_port": frontend_port if target in ("both", "all") else None,
                "lora_port": lora_port if target == "all" else None,
                "timeout": health_timeout,
                "interval": HEALTH_CHECK_INTERVAL,
                "stop_event": manager._stopping,
            },
            daemon=True,
            name="backend-health-probe",
        )
        health_thread.start()

    # Wait for completion or interrupt
    try:
        exit_code = manager.wait()
    finally:
        manager.stop_all(ports=target_ports)

    return exit_code


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog="launch.py",
        description="ARES Unified Terminal Launching Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python launch.py                    # Interactive terminal menu
  python launch.py both               # Launch Frontend & Backend concurrently
  python launch.py backend            # Launch FastAPI Backend on :8000
  python launch.py frontend           # Launch Blazor Web UI on :5149
  python launch.py all                # Launch Full Stack (Frontend + Backend + LoRA)
  python launch.py lora               # Launch LoRA Inference Server on :11435
  python launch.py status             # Inspect port & service status
  python launch.py clean              # Free all ARES ports (8000, 5149, 11435)
        """,
    )

    parser.add_argument(
        "target",
        nargs="?",
        default=None,
        choices=["backend", "frontend", "both", "all", "lora", "status", "clean", "kill"],
        help="Service target to launch (or action to perform)",
    )

    parser.add_argument("--backend-port", "-bp", type=int, default=DEFAULT_BACKEND_PORT, help="FastAPI port (default: 8000)")
    parser.add_argument("--frontend-port", "-fp", type=int, default=DEFAULT_FRONTEND_PORT, help="Blazor port (default: 5149)")
    parser.add_argument("--lora-port", "-lp", type=int, default=DEFAULT_LORA_PORT, help="LoRA port (default: 11435)")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Backend bind address (default: 127.0.0.1)")
    parser.add_argument("--no-reload", action="store_true", help="Disable FastAPI uvicorn auto-reloading")
    parser.add_argument("--skip-health-check", action="store_true", help="Skip backend readiness probe")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI color output")
    parser.add_argument("--no-clean", action="store_true", help="Do not auto-clean occupied ports before starting")
    parser.add_argument("--timeout", type=float, default=HEALTH_CHECK_TIMEOUT, help="Health check timeout in seconds")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Main CLI entrypoint."""
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    if hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    init_windows_vt_mode()
    args = parse_args(argv)

    if args.no_color or os.environ.get("NO_COLOR"):
        set_colors_enabled(False)

    env_info = detect_environment()

    # Route interactive vs direct invocation
    target = args.target
    if target is None:
        target = show_interactive_menu(env_info)

    if target in ("exit", "quit"):
        print("Goodbye!")
        return 0

    if target == "status":
        display_ascii_banner(env_info)
        check_ports_status([
            ("Backend", args.backend_port),
            ("Frontend", args.frontend_port),
            ("LoRA Server", args.lora_port),
        ])
        return 0

    if target in ("clean", "kill"):
        print(colorize("Freeing ARES platform ports...", Colors.BOLD))
        clean_ports([args.backend_port, args.frontend_port, args.lora_port])
        return 0

    try:
        return launch_engine(
            target=target,
            env_info=env_info,
            backend_port=args.backend_port,
            frontend_port=args.frontend_port,
            lora_port=args.lora_port,
            host=args.host,
            reload=not args.no_reload,
            skip_health_check=args.skip_health_check,
            auto_clean_ports=not args.no_clean,
            health_timeout=args.timeout,
        )
    except RuntimeError as e:
        log_message("ERROR", str(e), Colors.RED)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
