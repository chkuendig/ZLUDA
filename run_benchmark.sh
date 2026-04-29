#!/bin/bash
# Deploy current ZLUDA build, launch Batman benchmark, capture screenshots on
# key events, and archive logs + captures to a timestamped directory.
#
# This is a thin wrapper around deploy_batman.sh that adds:
#  - Screenshot at benchmark start (via spectacle)
#  - Screenshot on crash (if SIGABRT/SIGSEGV marker is detected in the log)
#  - Automatic archival of tex log, debug log, screenshots, and /tmp/*_captures.bin
#    files into ~/Desktop/zluda/run_archives/run_<timestamp>/
#
# Usage:
#   ./run_benchmark.sh [--tag NAME] [-- <args passed to deploy_batman.sh>]
#
# Examples:
#   ./run_benchmark.sh                        # build, deploy, run, archive
#   ./run_benchmark.sh --tag bbox_fix         # add tag to archive dir name
#   ./run_benchmark.sh -- --no-build          # skip the build step
#
# Environment overrides:
#   ARCHIVE_ROOT     Where run_<timestamp> dirs are created
#                    (default: ~/Desktop/zluda/run_archives)
#   SCREENSHOT_CMD   Command to use for screenshots
#                    (default: spectacle -b -n -f -o <path>)
#   LOG_FILE         ZLUDA tex log (default: /tmp/zluda_tex.log)
#   DEBUG_LOG_FILE   ZLUDA debug log (default: /tmp/zluda_debug.log)
#   KERNEL_START_THRESHOLD  How many cuLaunchKernel calls indicate benchmark start
#                           (default: 200)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_SCRIPT="$SCRIPT_DIR/deploy_batman.sh"

ARCHIVE_ROOT="${ARCHIVE_ROOT:-$HOME/Desktop/zluda/run_archives}"
LOG_FILE="${LOG_FILE:-/tmp/zluda_tex.log}"
DEBUG_LOG_FILE="${DEBUG_LOG_FILE:-/tmp/zluda_debug.log}"
STEAM_LOG_FILE="/tmp/batman_steam_launch.log"
KERNEL_START_THRESHOLD="${KERNEL_START_THRESHOLD:-200}"

TAG=""
DEPLOY_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tag)
            TAG="$2"
            shift 2
            ;;
        --)
            shift
            DEPLOY_ARGS+=("$@")
            break
            ;;
        -h|--help)
            sed -n '2,30p' "$0"
            exit 0
            ;;
        *)
            DEPLOY_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ ! -f "$DEPLOY_SCRIPT" ]]; then
    echo "Error: deploy_batman.sh not found at $DEPLOY_SCRIPT" >&2
    exit 1
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
ARCHIVE_NAME="run_$TIMESTAMP"
if [[ -n "$TAG" ]]; then
    ARCHIVE_NAME="${ARCHIVE_NAME}_${TAG}"
fi
ARCHIVE_DIR="$ARCHIVE_ROOT/$ARCHIVE_NAME"
mkdir -p "$ARCHIVE_DIR"

log() { echo "[run_benchmark] $*" >&2; }

log "archive dir: $ARCHIVE_DIR"

# -- X session discovery for spectacle ---------------------------------------
# Reuse the same heuristic deploy_batman.sh uses: search for a kwin/plasma
# process and pick up its DISPLAY / XAUTHORITY. deploy_batman.sh already
# handles xdotool; spectacle needs the same env.
discover_display() {
    local pid disp auth
    for pid in $(pgrep -f 'kwin_x11|plasmashell|startplasma-x11' 2>/dev/null); do
        [[ -r "/proc/$pid/environ" ]] || continue
        disp="$(tr '\0' '\n' < "/proc/$pid/environ" | sed -n 's/^DISPLAY=//p' | head -n1)"
        auth="$(tr '\0' '\n' < "/proc/$pid/environ" | sed -n 's/^XAUTHORITY=//p' | head -n1)"
        if [[ -n "$disp" ]]; then
            export DISPLAY="$disp"
            if [[ -n "$auth" && -r "$auth" ]]; then
                export XAUTHORITY="$auth"
            fi
            return 0
        fi
    done
    # Fallback: assume :0 with standard xauth
    if [[ -S /tmp/.X11-unix/X0 ]]; then
        export DISPLAY=":0"
        local auth
        for auth in /run/user/"$(id -u)"/xauth_* "$HOME/.Xauthority"; do
            if [[ -r "$auth" ]]; then
                export XAUTHORITY="$auth"
                return 0
            fi
        done
        return 0
    fi
    return 1
}

take_screenshot() {
    local out="$1"
    local label="$2"
    if ! command -v spectacle >/dev/null 2>&1; then
        log "screenshot skipped: spectacle not installed ($label)"
        return 0
    fi
    if ! discover_display; then
        log "screenshot skipped: no X session found ($label)"
        return 0
    fi
    if spectacle -b -n -f -o "$out" >/dev/null 2>&1; then
        log "screenshot saved: $out ($label)"
    else
        log "screenshot failed ($label)"
    fi
}

# -- Event watcher -----------------------------------------------------------
# Runs in a subshell, watches the tex log for benchmark-start and crash
# markers, fires take_screenshot on each.
#
# Important: the backgrounded subshell's stdout is redirected to /dev/null so
# that `start_event_watcher` can be called via command substitution without
# the subshell keeping the capture pipe open (which would hang the caller).
start_event_watcher() {
    local log_file="$1"
    local archive_dir="$2"
    local threshold="$3"
    (
        local saw_start=0
        local saw_crash=0
        local initial_lines
        if [[ -f "$log_file" ]]; then
            initial_lines="$(wc -l < "$log_file")"
        else
            initial_lines=0
        fi
        while true; do
            sleep 5
            if [[ ! -f "$log_file" ]]; then
                continue
            fi
            local new_lines
            new_lines="$(tail -n +$((initial_lines + 1)) "$log_file" 2>/dev/null)"
            if (( saw_start == 0 )); then
                local n_launches
                n_launches="$(printf '%s\n' "$new_lines" | grep -c 'cuLaunchKernel' || true)"
                if (( n_launches >= threshold )); then
                    saw_start=1
                    take_screenshot "$archive_dir/screenshot_benchmark_start.png" "benchmark-start"
                fi
            fi
            if (( saw_crash == 0 )); then
                if printf '%s\n' "$new_lines" | grep -qE '\*\*\* SIGABRT|GPU fault likely|\*\*\* SIGSEGV'; then
                    saw_crash=1
                    take_screenshot "$archive_dir/screenshot_crash.png" "crash"
                fi
            fi
            if (( saw_start == 1 && saw_crash == 1 )); then
                break
            fi
        done
    ) >/dev/null 2>&1 &
    echo "$!"
}

# -- Archive step ------------------------------------------------------------
archive_run() {
    local exit_code="$1"
    log "archiving run (exit=$exit_code)"
    [[ -f "$LOG_FILE" ]] && cp -a "$LOG_FILE" "$ARCHIVE_DIR/zluda_tex.log" 2>/dev/null || true
    [[ -f "$DEBUG_LOG_FILE" ]] && cp -a "$DEBUG_LOG_FILE" "$ARCHIVE_DIR/zluda_debug.log" 2>/dev/null || true
    [[ -f "$STEAM_LOG_FILE" ]] && cp -a "$STEAM_LOG_FILE" "$ARCHIVE_DIR/batman_steam_launch.log" 2>/dev/null || true
    # /tmp capture files: ms4_captures.bin, ms2b_captures.bin, tfs_captures.bin,
    # iofx_bbox_captures.bin, zluda_sort_replay_capture_*.bin
    shopt -s nullglob
    for f in /tmp/ms4_captures.bin /tmp/ms2b_captures.bin /tmp/tfs_captures.bin \
             /tmp/iofx_bbox_captures.bin /tmp/iofx_captures.bin \
             /tmp/zluda_sort_replay_capture_*.bin; do
        [[ -f "$f" ]] || continue
        cp -a "$f" "$ARCHIVE_DIR/" 2>/dev/null || true
    done
    shopt -u nullglob

    # Write a short summary for future inspection.
    {
        echo "run_benchmark archive"
        echo "timestamp:     $TIMESTAMP"
        echo "tag:           ${TAG:-(none)}"
        echo "deploy_args:   ${DEPLOY_ARGS[*]:-(none)}"
        echo "exit_code:     $exit_code"
        if [[ -f "$ARCHIVE_DIR/zluda_tex.log" ]]; then
            echo "tex_log_lines: $(wc -l < "$ARCHIVE_DIR/zluda_tex.log")"
            local last_ts
            last_ts="$(grep -oE '\[[0-9]+ms\]' "$ARCHIVE_DIR/zluda_tex.log" | tail -n1 | tr -d '[]ms')"
            [[ -n "$last_ts" ]] && echo "last_log_ms:   $last_ts"
        fi

        # Phase 1a (2026-04-28): run-sequence + outcome classification.
        # Counts of session boundaries and benchmark starts in the tex log.
        # A "benchmark started" line is emitted by deploy_batman.sh when
        # cuLaunchKernel ≥ KERNEL_START_THRESHOLD. If we see >1, the user
        # (or harness) re-entered the benchmark within the same Batman
        # session — relevant for the A vs B disambiguation.
        local n_zluda_init n_bench_started n_apply_forces
        n_zluda_init=$(grep -c '=== ZLUDA init' "$ARCHIVE_DIR/zluda_tex.log" 2>/dev/null || echo 0)
        # No ZLUDA-side marker for benchmark re-entry; instead, count
        # "Title prompt advanced" / explicit harness markers if present.
        # As a proxy, count clusters of cuLaunchKernel'CreateParticleShapesIncremental_MultiStage_1'
        # which fire at every benchmark scene init.
        n_bench_started=$(grep -c "cuLaunchKernel 'CreateParticleShapesIncremental_MultiStage_1'" "$ARCHIVE_DIR/zluda_tex.log" 2>/dev/null || echo 0)
        n_apply_forces=$(grep -c "cuLaunchKernel 'TurbulenceFS_dynamicGridApplyForcesKernel'" "$ARCHIVE_DIR/zluda_tex.log" 2>/dev/null || echo 0)
        echo "zluda_init_count: $n_zluda_init"
        echo "ms1_kernel_count: $n_bench_started"
        echo "apply_forces:     $n_apply_forces"

        # Crash detection + signature classification.
        local crash_kind="none"
        if grep -qE '\*\*\* SIGABRT|GPU fault likely|\*\*\* SIGSEGV' "$ARCHIVE_DIR/zluda_tex.log" 2>/dev/null; then
            if grep -qE '\*\*\* SIGABRT' "$ARCHIVE_DIR/zluda_tex.log"; then
                crash_kind="SIGABRT"
            elif grep -qE '\*\*\* SIGSEGV' "$ARCHIVE_DIR/zluda_tex.log"; then
                crash_kind="SIGSEGV"
            else
                crash_kind="signal"
            fi
            echo "crash:         yes ($crash_kind)"
            grep -E '\*\*\* SIGABRT|GPU fault likely|\*\*\* SIGSEGV' "$ARCHIVE_DIR/zluda_tex.log" | head -3

            # Outcome heuristic — Phase 1a:
            # - benchmark_count=0  → pre_bench_crash (died during init/menu)
            # - benchmark_count=1  → mid_or_initial   (one benchmark run, then crash — could be mid-run OR re-entry crash before second MS1)
            # - benchmark_count>=2 → reentry_crash   (multi-bench session, crash after re-entry)
            local outcome
            if (( n_bench_started == 0 )); then
                outcome="pre_benchmark_crash"
            elif (( n_bench_started >= 2 )); then
                outcome="post_reentry_crash"
            elif (( n_apply_forces > 100 )); then
                outcome="mid_run_crash_after_AF"
            else
                outcome="early_run_crash"
            fi
            echo "outcome:       $outcome"
        else
            echo "crash:         no"
            # No crash detected. Check whether benchmark completed normally
            # by looking at the last log timestamp gap and kernel activity.
            # If last_log_ms > 600000 (10 min) and no crash, very likely
            # the benchmark completed and the harness timed out monitoring.
            local last_ms_int
            last_ms_int=$(grep -oE '\[[0-9]+ms\]' "$ARCHIVE_DIR/zluda_tex.log" 2>/dev/null | tail -n1 | tr -d '[]ms')
            if [[ -n "$last_ms_int" && "$last_ms_int" -gt 600000 ]]; then
                echo "outcome:       completed_or_idle"
            else
                echo "outcome:       early_exit"
            fi
        fi

        # Latest BatmanAK coredump (matched by recency, not pid).
        local latest_core
        latest_core=$(coredumpctl list --since "today" 2>/dev/null | grep -i BatmanAK | tail -1 | awk '{print $5}')
        if [[ -n "$latest_core" ]]; then
            echo "latest_core_pid: $latest_core"
            # Resolve crash library (PhysX / HIP / zluda Rust / other).
            local crash_lib
            crash_lib=$(coredumpctl info "$latest_core" 2>/dev/null | grep -oE 'libamdhip64\.so|PhysX3_x64\.dll|libnvcuda\.so|libc\.so|wine_stack_fix' | head -1)
            [[ -n "$crash_lib" ]] && echo "crash_library: $crash_lib"
        fi
    } > "$ARCHIVE_DIR/SUMMARY.txt"
    log "archive complete: $ARCHIVE_DIR"
    ls -la "$ARCHIVE_DIR" >&2 || true
}

# -- Run ---------------------------------------------------------------------
WATCHER_PID=""
cleanup() {
    local ec="${1:-$?}"
    if [[ -n "$WATCHER_PID" ]]; then
        kill "$WATCHER_PID" 2>/dev/null || true
        wait "$WATCHER_PID" 2>/dev/null || true
    fi
    archive_run "$ec"
    exit "$ec"
}
trap 'cleanup $?' EXIT
trap 'cleanup 130' INT TERM

WATCHER_PID="$(start_event_watcher "$LOG_FILE" "$ARCHIVE_DIR" "$KERNEL_START_THRESHOLD")"
log "event watcher pid: $WATCHER_PID"

log "invoking deploy_batman.sh ${DEPLOY_ARGS[*]:-}"
if bash "$DEPLOY_SCRIPT" "${DEPLOY_ARGS[@]}"; then
    RUN_EC=0
else
    RUN_EC=$?
    log "deploy_batman.sh returned exit=$RUN_EC"
fi

# cleanup trap fires on exit with $RUN_EC
exit "$RUN_EC"
