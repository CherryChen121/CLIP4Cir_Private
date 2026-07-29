#!/usr/bin/env bash
set -u

PROJECT_ROOT="/data0/qrchen/projects/CLIP4Cir"
PYTHON_BIN="/data0/qrchen/miniconda3/envs/clip4cir/bin/python"
RUNTIME_ROOT="${PROJECT_ROOT}/gpu_queue_runs"
DISPATCHER="${PROJECT_ROOT}/src/combined_gpu_queue.py"

usage() {
    echo "Usage: $0 dry-run | start | resume /absolute/path/to/run-dir" >&2
    exit 2
}

case "${1:-}" in
    dry-run)
        [ "$#" -eq 1 ] || usage
        cd "$PROJECT_ROOT" || exit 1
        exec "$PYTHON_BIN" "$DISPATCHER" --dry-run
        ;;
    start)
        [ "$#" -eq 1 ] || usage
        mkdir -p "$RUNTIME_ROOT" || exit 1
        launcher_log="${RUNTIME_ROOT}/launcher_$(date +%Y%m%d-%H%M%S-%N).log"
        cd "$PROJECT_ROOT" || exit 1
        nohup "$PYTHON_BIN" "$DISPATCHER" --run > "$launcher_log" 2>&1 &
        dispatcher_pid=$!
        echo "Dispatcher PID: ${dispatcher_pid}"
        echo "Launcher log: ${launcher_log}"
        ;;
    resume)
        [ "$#" -eq 2 ] || usage
        case "$2" in
            /*) run_dir="$2" ;;
            *) echo "resume requires an absolute run directory" >&2; exit 2 ;;
        esac
        [ -f "${run_dir}/state.json" ] || {
            echo "state.json not found in ${run_dir}" >&2
            exit 2
        }
        mkdir -p "$RUNTIME_ROOT" || exit 1
        launcher_log="${RUNTIME_ROOT}/launcher_resume_$(date +%Y%m%d-%H%M%S-%N).log"
        cd "$PROJECT_ROOT" || exit 1
        nohup "$PYTHON_BIN" "$DISPATCHER" --resume "$run_dir" > "$launcher_log" 2>&1 &
        dispatcher_pid=$!
        echo "Dispatcher PID: ${dispatcher_pid}"
        echo "Launcher log: ${launcher_log}"
        ;;
    *)
        usage
        ;;
esac
