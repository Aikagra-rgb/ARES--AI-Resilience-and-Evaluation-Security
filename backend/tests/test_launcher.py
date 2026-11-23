"""Unit tests for the ARES Unified Launching Terminal Engine (launch.py)."""

from __future__ import annotations

import argparse
import os
import socket
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add repository root to sys.path so launch can be imported
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import launch
from launch import (
    Colors,
    EnvironmentInfo,
    ProcessManager,
    ServiceTask,
    build_child_env,
    build_service_tasks,
    clean_ports,
    colorize,
    detect_environment,
    ensure_database_ready,
    get_child_pids,
    get_pids_for_port,
    is_port_in_use,
    kill_process_tree,
    launch_engine,
    main,
    parse_args,
    probe_backend_health,
    set_colors_enabled,
    show_interactive_menu,
)


def test_parse_args_defaults() -> None:
    args = parse_args([])
    assert args.target is None
    assert args.backend_port == 8000
    assert args.frontend_port == 5149
    assert args.targets_port == 8001
    assert args.lora_port == 11435
    assert args.host == "127.0.0.1"
    assert args.no_reload is False
    assert args.skip_health_check is False
    assert args.no_color is False
    assert args.no_clean is False
    assert args.mock_targets is False
    assert args.sqlite is False
    assert args.require_postgres is False
    assert args.timeout == 45.0


@pytest.mark.parametrize(
    "target_name",
    ["backend", "frontend", "both", "targets", "suite", "all", "lora", "status", "clean", "kill"],
)
def test_parse_args_valid_targets(target_name: str) -> None:
    args = parse_args([target_name])
    assert args.target == target_name


def test_parse_args_custom_ports_and_flags() -> None:
    args = parse_args([
        "suite",
        "--backend-port", "9000",
        "--frontend-port", "6000",
        "--targets-port", "8005",
        "--lora-port", "12000",
        "--host", "0.0.0.0",
        "--no-reload",
        "--skip-health-check",
        "--no-color",
        "--no-clean",
        "--mock-targets",
        "--timeout", "60",
    ])
    assert args.target == "suite"
    assert args.backend_port == 9000
    assert args.frontend_port == 6000
    assert args.targets_port == 8005
    assert args.lora_port == 12000
    assert args.host == "0.0.0.0"
    assert args.no_reload is True
    assert args.skip_health_check is True
    assert args.no_color is True
    assert args.no_clean is True
    assert args.mock_targets is True
    assert args.timeout == 60.0


def test_parse_args_short_port_flags() -> None:
    args = parse_args(["all", "-bp", "8080", "-fp", "5000", "-tp", "8002", "-lp", "11000"])
    assert args.backend_port == 8080
    assert args.frontend_port == 5000
    assert args.targets_port == 8002
    assert args.lora_port == 11000


def test_colorize_toggle() -> None:
    set_colors_enabled(True)
    colored = colorize("ARES", Colors.RED)
    assert Colors.RED in colored
    assert Colors.RESET in colored

    set_colors_enabled(False)
    plain = colorize("ARES", Colors.RED)
    assert plain == "ARES"
    set_colors_enabled(True)


def test_detect_environment(tmp_path: Path) -> None:
    env_info = detect_environment(repo_root=repo_root)
    assert env_info.repo_root == repo_root
    assert env_info.backend_dir == repo_root / "backend"
    assert env_info.frontend_dir == repo_root / "src" / "Ares.Web"
    assert env_info.python_path.exists()
    assert env_info.python_version is not None


def test_build_child_env() -> None:
    env_info = detect_environment(repo_root=repo_root)
    child_env = build_child_env(env_info)

    assert child_env["PYTHONUNBUFFERED"] == "1"
    assert "PYTHONPATH" in child_env
    assert str(env_info.backend_dir) in child_env["PYTHONPATH"]
    if env_info.venv_path:
        assert child_env["VIRTUAL_ENV"] == str(env_info.venv_path)


def test_build_service_tasks_backend_only() -> None:
    env_info = detect_environment(repo_root=repo_root)
    tasks = build_service_tasks("backend", env_info, backend_port=8000, host="127.0.0.1", reload=True)

    assert len(tasks) == 1
    task = tasks[0]
    assert task.tag == "BACKEND"
    assert task.port == 8000
    assert task.cwd == env_info.backend_dir
    assert "-m" in task.command
    assert "uvicorn" in task.command
    assert "app.main:app" in task.command
    assert "--reload" in task.command
    assert "8000" in task.command


def test_build_service_tasks_both() -> None:
    env_info = detect_environment(repo_root=repo_root)
    # Ensure dotnet_path is mocked if not installed on machine
    if not env_info.dotnet_path:
        env_info.dotnet_path = Path("/mock/dotnet")

    tasks = build_service_tasks("both", env_info, backend_port=8000, frontend_port=5149, reload=False)

    assert len(tasks) == 2
    tags = [t.tag for t in tasks]
    assert "BACKEND" in tags
    assert "FRONTEND" in tags

    backend_task = next(t for t in tasks if t.tag == "BACKEND")
    assert "--reload" not in backend_task.command

    frontend_task = next(t for t in tasks if t.tag == "FRONTEND")
    assert frontend_task.port == 5149
    assert "run" in frontend_task.command
    assert "--project" in frontend_task.command
    assert "http://localhost:5149" in frontend_task.command[frontend_task.command.index("--urls") + 1]


def test_build_service_tasks_all() -> None:
    env_info = detect_environment(repo_root=repo_root)
    if not env_info.dotnet_path:
        env_info.dotnet_path = Path("/mock/dotnet")

    tasks = build_service_tasks(
        "all",
        env_info,
        backend_port=8000,
        frontend_port=5149,
        lora_port=11435,
        targets_port=8001,
    )

    assert len(tasks) == 4
    tags = [t.tag for t in tasks]
    assert "BACKEND" in tags
    assert "FRONTEND" in tags
    assert "TARGETS" in tags
    assert "LORA" in tags

    targets_task = next(t for t in tasks if t.tag == "TARGETS")
    assert targets_task.port == 8001
    assert "tests.targets:app" in targets_task.command
    assert targets_task.env["ARES_GATEWAY_URL"] == "http://127.0.0.1:8000"

    lora_task = next(t for t in tasks if t.tag == "LORA")
    assert lora_task.port == 11435
    assert "ares.training.serve_lora" in lora_task.command
    assert "11435" in lora_task.command


def test_build_service_tasks_targets_and_suite() -> None:
    env_info = detect_environment(repo_root=repo_root)
    if not env_info.dotnet_path:
        env_info.dotnet_path = Path("/mock/dotnet")

    targets_only = build_service_tasks(
        "targets",
        env_info,
        backend_port=8100,
        targets_port=8101,
        reload=False,
        mock_targets_llm=True,
    )
    assert len(targets_only) == 1
    t_task = targets_only[0]
    assert t_task.tag == "TARGETS"
    assert t_task.port == 8101
    assert "tests.targets:app" in t_task.command
    assert "--reload" not in t_task.command
    assert t_task.env["ARES_GATEWAY_URL"] == "http://127.0.0.1:8100"
    assert t_task.env["ARES_TARGET_MOCK_LLM"] == "1"

    suite_tasks = build_service_tasks(
        "suite",
        env_info,
        backend_port=8000,
        frontend_port=5149,
        targets_port=8001,
    )
    assert len(suite_tasks) == 3
    assert [t.tag for t in suite_tasks] == ["BACKEND", "FRONTEND", "TARGETS"]


def test_build_service_tasks_frontend_missing_dotnet_raises() -> None:
    env_info = EnvironmentInfo(
        repo_root=repo_root,
        backend_dir=repo_root / "backend",
        frontend_dir=repo_root / "src" / "Ares.Web",
        python_path=Path(sys.executable),
        dotnet_path=None,
        venv_path=None,
        is_windows=sys.platform == "win32",
    )

    with pytest.raises(RuntimeError, match=r"\.NET SDK CLI \('dotnet'\) was not found"):
        build_service_tasks("frontend", env_info)


def test_is_port_in_use() -> None:
    # Bind an ephemeral port to test detection
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    port = sock.getsockname()[1]

    try:
        assert is_port_in_use(port) is True
    finally:
        sock.close()

    # Free port should not be in use
    assert is_port_in_use(port) is False


def test_clean_ports_empty() -> None:
    # Non-existent or free port should return 0 killed
    killed = clean_ports([49151], verbose=False)
    assert killed == 0


def test_kill_process_tree_invalid_pid() -> None:
    assert kill_process_tree(-1) is False
    assert kill_process_tree(0) is False


def test_probe_backend_health_success() -> None:
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        ready = probe_backend_health(
            health_url="http://127.0.0.1:8000/health",
            backend_port=8000,
            timeout=1.0,
            interval=0.1,
        )
        assert ready is True


def test_probe_backend_health_timeout() -> None:
    with patch("urllib.request.urlopen", side_effect=OSError("Connection refused")):
        ready = probe_backend_health(
            health_url="http://127.0.0.1:8000/health",
            backend_port=8000,
            timeout=0.3,
            interval=0.1,
        )
        assert ready is False


def test_process_manager_stop_all_idempotency() -> None:
    manager = ProcessManager()
    mock_task = ServiceTask(
        name="MockService",
        tag="MOCK",
        color=Colors.CYAN,
        command=["python", "-c", "import time; time.sleep(1)"],
        cwd=repo_root,
        env={},
        port=8000,
    )
    mock_proc = MagicMock()
    mock_proc.pid = 99999
    mock_proc.poll.return_value = 0
    manager.processes.append((mock_task, mock_proc))

    with patch("launch.kill_process_tree") as mock_kill, patch("launch.clean_ports") as mock_clean:
        # First call triggers stop
        manager.stop_all(ports=[8000])
        # Second call should be a no-op due to idempotency guard
        manager.stop_all(ports=[8000])

        assert manager._stopped is True
        assert mock_clean.call_count == 1


def test_get_child_pids_invalid() -> None:
    assert get_child_pids(-1) == []
    assert get_child_pids(0) == []


def test_get_pids_for_port_windows_netstat_mock() -> None:
    mock_netstat_output = """
Active Connections

  Proto  Local Address          Foreign Address        State           PID
  TCP    0.0.0.0:800            0.0.0.0:0              LISTENING       1111
  TCP    0.0.0.0:8000           0.0.0.0:0              LISTENING       2222
  TCP    127.0.0.1:8000         0.0.0.0:0              LISTENING       2222
  TCP    10.0.0.1:55432         10.0.0.2:8000          ESTABLISHED     3333
  TCP    0.0.0.0:80000          0.0.0.0:0              LISTENING       4444
  TCP    [::]:8000              [::]:0                 LISTENING       5555
"""
    mock_res = MagicMock()
    mock_res.returncode = 0
    mock_res.stdout = mock_netstat_output

    with patch("sys.platform", "win32"), patch("subprocess.run", return_value=mock_res):
        pids = get_pids_for_port(8000)
        # Should only match PID 2222 and 5555, NOT 1111 (port 800), 3333 (foreign addr), or 4444 (port 80000)
        assert 2222 in pids
        assert 5555 in pids
        assert 1111 not in pids
        assert 3333 not in pids
        assert 4444 not in pids


def test_build_service_tasks_frontend_backend_env_propagation() -> None:
    env_info = detect_environment(repo_root=repo_root)
    if not env_info.dotnet_path:
        env_info.dotnet_path = Path("/mock/dotnet")

    tasks = build_service_tasks(
        "both",
        env_info,
        backend_port=9090,
        frontend_port=5200,
        reload=False,
    )

    frontend_task = next(t for t in tasks if t.tag == "FRONTEND")
    assert frontend_task.port == 5200
    assert frontend_task.env["Ares__ApiBaseUrl"] == "http://127.0.0.1:9090"
    assert "http://localhost:5200" in frontend_task.env["ASPNETCORE_URLS"]


def test_launch_engine_aborts_if_port_remains_occupied() -> None:
    env_info = detect_environment(repo_root=repo_root)
    with patch("launch.is_port_in_use", return_value=True), patch("launch.clean_ports"):
        code = launch_engine(
            target="backend",
            env_info=env_info,
            backend_port=8000,
            auto_clean_ports=True,
        )
        assert code == 1


def test_show_interactive_menu_selection() -> None:
    env_info = detect_environment(repo_root=repo_root)
    with patch("builtins.input", side_effect=["1"]), patch("launch.check_ports_status"):
        choice = show_interactive_menu(env_info)
        assert choice == "both"

    with patch("builtins.input", side_effect=["backend"]), patch("launch.check_ports_status"):
        choice = show_interactive_menu(env_info)
        assert choice == "backend"

    with patch("builtins.input", side_effect=EOFError), patch("launch.check_ports_status"):
        choice = show_interactive_menu(env_info)
        assert choice == "exit"


def test_main_status_and_clean_routing() -> None:
    with patch("launch.check_ports_status") as mock_status:
        code = main(["status", "--no-color"])
        assert code == 0
        assert mock_status.called

    with patch("launch.clean_ports") as mock_clean:
        code = main(["clean", "--no-color"])
        assert code == 0
        assert mock_clean.called


def test_parse_args_database_flags() -> None:
    sqlite_args = parse_args(["suite", "--sqlite"])
    assert sqlite_args.sqlite is True
    assert sqlite_args.require_postgres is False

    pg_args = parse_args(["suite", "--require-postgres"])
    assert pg_args.require_postgres is True
    assert pg_args.sqlite is False


def test_ensure_database_ready_postgres_reachable() -> None:
    env_info = detect_environment(repo_root=repo_root)
    with (
        patch("launch.read_configured_database_url", return_value="postgresql+psycopg://ares:pw@127.0.0.1:5432/ares"),
        patch("launch._is_tcp_reachable", return_value=True),
    ):
        db_env, db_label = ensure_database_ready(env_info)
    assert db_env == {}
    assert "PostgreSQL" in db_label


def test_ensure_database_ready_auto_starts_docker_postgres() -> None:
    env_info = detect_environment(repo_root=repo_root)
    with (
        patch("launch.read_configured_database_url", return_value="postgresql+psycopg://ares:pw@127.0.0.1:5432/ares"),
        patch("launch._is_tcp_reachable", return_value=False),
        patch("launch._try_start_docker_postgres", return_value=True),
    ):
        db_env, db_label = ensure_database_ready(env_info)
    assert db_env == {}
    assert "Docker" in db_label


def test_ensure_database_ready_falls_back_to_sqlite_when_postgres_and_docker_down() -> None:
    env_info = detect_environment(repo_root=repo_root)
    with (
        patch("launch.read_configured_database_url", return_value="postgresql+psycopg://ares:pw@127.0.0.1:5432/ares"),
        patch("launch._is_tcp_reachable", return_value=False),
        patch("launch._try_start_docker_postgres", return_value=False),
    ):
        db_env, db_label = ensure_database_ready(env_info)
    assert db_env["DATABASE_URL"].startswith("sqlite:///")
    assert db_env["AUTO_CREATE_SCHEMA"] == "true"
    assert db_env["ENVIRONMENT"] == "development"
    assert "SQLite" in db_label


def test_ensure_database_ready_require_postgres_raises() -> None:
    env_info = detect_environment(repo_root=repo_root)
    with (
        patch("launch.read_configured_database_url", return_value="postgresql+psycopg://ares:pw@127.0.0.1:5432/ares"),
        patch("launch._is_tcp_reachable", return_value=False),
        patch("launch._try_start_docker_postgres", return_value=False),
    ):
        with pytest.raises(RuntimeError, match="PostgreSQL is unreachable"):
            ensure_database_ready(env_info, require_postgres=True)


def test_config_allows_sqlite_only_in_development() -> None:
    from ares.config import Settings

    dev_cfg = Settings(DATABASE_URL="sqlite:///test.db", ENVIRONMENT="development")
    assert dev_cfg.database_url == "sqlite:///test.db"

    with pytest.raises(ValueError, match="PostgreSQL"):
        Settings(
            DATABASE_URL="sqlite:///test.db",
            ENVIRONMENT="production",
            ARES_AUTH_ENABLED=True,
            ALLOW_DEV_IDENTITY=False,
            PROMPT_FINGERPRINT_SECRET="x" * 32,
            EXECUTION_PAYLOAD_ENCRYPTION_KEY="y" * 32,
            ARES_API_KEY="z" * 32,
            GATEWAY_CANARY_TOKEN="c" * 32,
        )


def test_build_child_env_sets_fortran_ctrl_handler() -> None:
    env_info = detect_environment(repo_root=repo_root)
    child_env = build_child_env(env_info)
    assert child_env["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] == "1"


def test_launch_engine_all_promotes_default_timeout_to_90s() -> None:
    env_info = detect_environment(repo_root=repo_root)
    captured_timeouts: list[float] = []

    def fake_probe(**kwargs):
        captured_timeouts.append(kwargs["timeout"])
        return True

    fake_manager = MagicMock()
    fake_manager.wait.return_value = 0

    with (
        patch("launch.is_port_in_use", return_value=False),
        patch("launch.ensure_database_ready", return_value=({}, "SQLite")),
        patch("launch.ProcessManager", return_value=fake_manager),
        patch("launch.probe_backend_health", side_effect=fake_probe),
        patch("launch.probe_targets_health", side_effect=fake_probe),
        patch("threading.Thread") as mock_thread_cls,
    ):
        def immediate_thread(*, target=None, kwargs=None, **rest):
            t = MagicMock()
            t.start.side_effect = lambda: target(**(kwargs or {}))
            return t

        mock_thread_cls.side_effect = immediate_thread
        code = launch_engine("all", env_info, health_timeout=45.0)
        assert code == 0
        assert captured_timeouts == [90.0, 90.0]

        captured_timeouts.clear()
        code2 = launch_engine("all", env_info, health_timeout=30.0)
        assert code2 == 0
        assert captured_timeouts == [30.0, 30.0]


def test_detect_environment_falls_back_on_ctypes_wdac_failure(tmp_path: Path) -> None:
    is_win = sys.platform == "win32"
    venv_dir = tmp_path / "venv"
    scripts_dir = venv_dir / ("Scripts" if is_win else "bin")
    scripts_dir.mkdir(parents=True)
    fake_venv_py = scripts_dir / ("python.exe" if is_win else "python")
    fake_venv_py.write_text("", encoding="utf-8")

    site_packages = (
        venv_dir / "Lib" / "site-packages"
        if is_win
        else venv_dir / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    )
    site_packages.mkdir(parents=True)

    base_executable = Path(sys.executable)
    (venv_dir / "pyvenv.cfg").write_text(f"executable = {base_executable}\n", encoding="utf-8")

    def fake_run(cmd, **_kwargs):
        proc_res = MagicMock()
        if str(cmd[0]) == str(fake_venv_py) and "-c" in cmd and "ctypes" in cmd[cmd.index("-c") + 1]:
            proc_res.returncode = 1
            proc_res.stdout = ""
            proc_res.stderr = "ImportError: DLL load failed while importing _ctypes"
            return proc_res
        if "--version" in cmd:
            proc_res.returncode = 0
            proc_res.stdout = f"Python {sys.version.split()[0]}"
            proc_res.stderr = ""
            return proc_res
        proc_res.returncode = 0
        proc_res.stdout = ""
        proc_res.stderr = ""
        return proc_res

    with patch("launch.subprocess.run", side_effect=fake_run):
        env_info = detect_environment(repo_root=tmp_path)

    assert env_info.python_path == base_executable
    assert env_info.venv_path == venv_dir

    child_env = build_child_env(env_info)
    assert str(site_packages) in child_env["PYTHONPATH"]
    assert child_env["VIRTUAL_ENV"] == str(venv_dir)



