@echo off
REM run.bat — Alpha Research 2026 startup script (Windows 11)
REM Usage: run.bat [command]
REM
REM Commands:
REM   backtest          Run full signal backtest (local, via Bazel)
REM   monitor           Run signal decay monitor (local, via Bazel)
REM   notebook          Launch JupyterLab (local)
REM   docker-build      Build Docker image (via Bazel)
REM   docker-notebook   Launch JupyterLab in Docker
REM   docker-backtest   Run backtest in Docker
REM   docker-down       Stop Docker stack
REM   test              Run all tests via Bazel
REM   build             Build all C++/Python targets via Bazel
setlocal EnableDelayedExpansion

set CMD=%~1
if "%CMD%"=="" set CMD=help

set BAZEL=bazel
set PYTHON=python

goto :%CMD% 2>nul || goto :unknown

:backtest
    echo [alpha-research] Running full backtest via Bazel...
    if "%N_DAYS%"=="" set N_DAYS=2520
    if "%N_CURRENCIES%"=="" set N_CURRENCIES=8
    %BAZEL% run //:backtest
    goto :eof

:monitor
    echo [alpha-research] Running signal decay monitor via Bazel...
    %BAZEL% run //:monitor
    goto :eof

:notebook
    echo [alpha-research] Installing Python package...
    pip install -e .[dev] --quiet
    echo [alpha-research] Launching JupyterLab...
    echo    Open: http://localhost:8888
    jupyter lab --notebook-dir=notebooks --ip=0.0.0.0 --no-browser
    goto :eof

:docker-build
    echo [alpha-research] Building Docker image via Bazel...
    %BAZEL% run //:docker_build
    goto :eof

:docker-notebook
    echo [alpha-research] Launching JupyterLab in Docker...
    echo    Open: http://localhost:8888
    docker compose up jupyter
    goto :eof

:docker-backtest
    echo [alpha-research] Running backtest in Docker...
    docker compose run --rm backtest
    goto :eof

:docker-down
    echo [alpha-research] Stopping Docker stack...
    %BAZEL% run //:docker_down
    goto :eof

:test
    echo [alpha-research] Running all tests (C++ + Python) via Bazel...
    %BAZEL% test //:test_all --test_output=short
    goto :eof

:build
    echo [alpha-research] Building all targets via Bazel...
    %BAZEL% build //...
    goto :eof

:unknown
:help
    echo.
    echo Alpha Research 2026 ^| run.bat
    echo.
    echo Usage: run.bat ^<command^>
    echo.
    echo Commands:
    echo   backtest         Full signal backtest (local, Bazel)
    echo   monitor          Signal decay monitor (local, Bazel)
    echo   notebook         JupyterLab (local)
    echo   docker-build     Build Docker image
    echo   docker-notebook  JupyterLab in Docker (http://localhost:8888)
    echo   docker-backtest  Backtest in Docker
    echo   docker-down      Stop Docker stack
    echo   test             All C++ + Python tests (Bazel)
    echo   build            Build all (Bazel)
    echo.
    echo Environment variables (set before calling):
    echo   set N_DAYS=2520
    echo   set N_CURRENCIES=8
    echo   set SLACK_WEBHOOK=https://hooks.slack.com/...
    echo.
    echo Examples:
    echo   run.bat backtest
    echo   run.bat docker-notebook
    echo   run.bat test
    goto :eof
