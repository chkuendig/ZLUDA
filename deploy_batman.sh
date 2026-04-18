#!/bin/bash
# Build, deploy, launch, and drive Batman: Arkham Knight into benchmark mode.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: ./deploy_batman.sh [options]

Options:
  --no-build           Skip the release build and deploy the existing libnvcuda.so
  --no-launch          Skip the Steam launch after deployment
  --no-benchmark       Launch only; do not wait for the menu or send benchmark keys
  --compare-captures   Inspect replay captures and run the offline good-vs-bad compare
  --watch-captures     Wait for the next good replay capture and run the offline compare automatically
  --watch-timeout SEC  Max seconds to wait in --watch-captures mode (default: 28800)
  --watch-interval SEC Poll interval in seconds for --watch-captures mode (default: 10)
  --good-capture PATH  Override the good replay capture path for --compare-captures
  --bad-capture PATH   Override the bad replay capture path for --compare-captures
  --compat-tool PATH   Steam compatibility tool root
  --steam-cli PATH     Steam CLI executable to use for launch
  --app-id ID          Steam app id to launch (default: 208650)
  --window-name NAME   Window title pattern to try to focus (default: Batman)
  --log-file PATH      ZLUDA log to monitor (default: /tmp/zluda_tex.log)
  -h, --help           Show this help text
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ZLUDA_DIR="$REPO_DIR/zluda"
COMPAT_TOOL_DIR="${HOME}/.local/share/Steam/compatibilitytools.d/proton-cuda"
STEAM_CLI="${STEAM_CLI:-steam}"
APP_ID="${STEAM_APPID:-208650}"
WINDOW_NAME="${BATMAN_WINDOW_NAME:-Batman}"
LOG_FILE="${BATMAN_LOG_FILE:-/tmp/zluda_tex.log}"
GAME_PROCESS_PATTERN="BatmanAK.exe -nologo"
DO_BUILD=1
DO_LAUNCH=1
DO_BENCHMARK=1
DO_COMPARE_CAPTURES=0
DO_WATCH_CAPTURES=0
GOOD_CAPTURE_PATH=""
BAD_CAPTURE_PATH=""
WATCH_TIMEOUT=28800
WATCH_INTERVAL=10

GAME_TIMEOUT=240
MENU_TIMEOUT=360
BENCHMARK_START_TIMEOUT=240
BENCHMARK_MONITOR_TIMEOUT=900
INITIAL_MENU_SETTLE=10
INITIAL_KEYPRESS_EXTRA_DELAY=20
MENU_VISIBLE_STABLE_SECONDS=10
MENU_IDLE_CALL_THRESHOLD=50
FOCUS_SETTLE_DELAY=0.2
BENCHMARK_RETRY_AFTER_SECONDS=20

X_DISPLAY="${DISPLAY-}"
X_AUTHORITY="${XAUTHORITY-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-build)
            DO_BUILD=0
            ;;
        --no-launch)
            DO_LAUNCH=0
            ;;
        --no-benchmark)
            DO_BENCHMARK=0
            ;;
        --compare-captures)
            DO_COMPARE_CAPTURES=1
            ;;
        --watch-captures)
            DO_COMPARE_CAPTURES=1
            DO_WATCH_CAPTURES=1
            ;;
        --watch-timeout)
            WATCH_TIMEOUT="$2"
            shift
            ;;
        --watch-interval)
            WATCH_INTERVAL="$2"
            shift
            ;;
        --good-capture)
            GOOD_CAPTURE_PATH="$2"
            shift
            ;;
        --bad-capture)
            BAD_CAPTURE_PATH="$2"
            shift
            ;;
        --compat-tool)
            COMPAT_TOOL_DIR="$2"
            shift
            ;;
        --steam-cli)
            STEAM_CLI="$2"
            shift
            ;;
        --app-id)
            APP_ID="$2"
            shift
            ;;
        --window-name)
            WINDOW_NAME="$2"
            shift
            ;;
        --log-file)
            LOG_FILE="$2"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Error: unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
    shift
done

LIBCUDA_SRC="$ZLUDA_DIR/target/release/libnvcuda.so"
PROTON_LIB64_DIR="$COMPAT_TOOL_DIR/files/lib64"
WINE_UNIX_DIR="$PROTON_LIB64_DIR/wine/x86_64-unix"
PROTONFIX_SRC="$REPO_DIR/protonfixes/gamefixes-steam/208650.py"
PROTONFIX_DST="$COMPAT_TOOL_DIR/protonfixes/gamefixes-steam/208650.py"
PROTONFIX_PKG_DST="$COMPAT_TOOL_DIR/protonfixes/protonfixes/gamefixes-steam/208650.py"
COMPUTE_CACHE_DB="${HOME}/.cache/zluda/ComputeCache/zluda.db"

status() {
    echo "[batman] $*" >&2
}

require_dir() {
    local path="$1"
    if [[ ! -d "$path" ]]; then
        echo "Error: directory not found: $path" >&2
        exit 1
    fi
}

require_file() {
    local path="$1"
    if [[ ! -f "$path" ]]; then
        echo "Error: file not found: $path" >&2
        exit 1
    fi
}

require_cmd() {
    local cmd="$1"
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "Error: required command not found: $cmd" >&2
        exit 1
    fi
}

install_if_changed() {
    local mode="$1"
    local src="$2"
    local dst="$3"
    if [[ -f "$dst" ]] && cmp -s "$src" "$dst"; then
        return
    fi
    install -D -m "$mode" "$src" "$dst"
}

log_cursor() {
    if [[ -f "$LOG_FILE" ]]; then
        printf '%s:%s\n' "$(stat -c %i "$LOG_FILE")" "$(wc -l < "$LOG_FILE")"
    else
        echo "missing:0"
    fi
}

print_new_log_lines() {
    local cursor="$1"
    if [[ ! -f "$LOG_FILE" ]]; then
        return
    fi
    local cursor_inode="${cursor%%:*}"
    local cursor_line="${cursor#*:}"
    local current_inode=""
    local current_line_count=0
    current_inode="$(stat -c %i "$LOG_FILE")"
    current_line_count="$(wc -l < "$LOG_FILE")"
    if [[ "$cursor_inode" != "$current_inode" || "$current_line_count" -lt "$cursor_line" ]]; then
        cat "$LOG_FILE"
    else
        tail -n +"$((cursor_line + 1))" "$LOG_FILE"
    fi
}

count_new_log_matches() {
    local cursor="$1"
    local pattern="$2"
    if [[ ! -f "$LOG_FILE" ]]; then
        echo 0
        return
    fi
    print_new_log_lines "$cursor" | grep -E -c "$pattern" || true
}

latest_new_log_match() {
    local cursor="$1"
    local pattern="$2"
    if [[ ! -f "$LOG_FILE" ]]; then
        return 1
    fi
    print_new_log_lines "$cursor" | grep -E "$pattern" | tail -n 1
}

game_running() {
    ps -eo cmd | grep -F "$GAME_PROCESS_PATTERN" | grep -v grep >/dev/null
}

discover_x_session() {
    local pid=""
    local disp=""
    local auth=""
    if [[ -n "$X_DISPLAY" && -n "$X_AUTHORITY" && -r "$X_AUTHORITY" ]]; then
        export DISPLAY="$X_DISPLAY"
        export XAUTHORITY="$X_AUTHORITY"
        return 0
    fi
    while read -r pid _; do
        [[ -r "/proc/$pid/environ" ]] || continue
        disp="$(tr '\0' '\n' < "/proc/$pid/environ" | sed -n 's/^DISPLAY=//p' | head -n 1)"
        auth="$(tr '\0' '\n' < "/proc/$pid/environ" | sed -n 's/^XAUTHORITY=//p' | head -n 1)"
        if [[ -n "$disp" && -n "$auth" && -r "$auth" && -S "/tmp/.X11-unix/X${disp#:}" ]]; then
            X_DISPLAY="$disp"
            X_AUTHORITY="$auth"
            export DISPLAY="$X_DISPLAY"
            export XAUTHORITY="$X_AUTHORITY"
            status "using X session DISPLAY=$DISPLAY XAUTHORITY=$XAUTHORITY"
            return 0
        fi
    done < <(ps -eo pid,cmd | grep -E 'kwin_x11|plasmashell|startplasma-x11' | grep -v grep)

    shopt -s nullglob
    local fallback_auth=(/run/user/"$(id -u)"/xauth_*)
    shopt -u nullglob
    if [[ -S /tmp/.X11-unix/X0 && ${#fallback_auth[@]} -gt 0 ]]; then
        X_DISPLAY=":0"
        X_AUTHORITY="${fallback_auth[0]}"
        export DISPLAY="$X_DISPLAY"
        export XAUTHORITY="$X_AUTHORITY"
        status "using fallback X session DISPLAY=$DISPLAY XAUTHORITY=$XAUTHORITY"
        return 0
    fi
    echo "Error: could not discover a usable X session for xdotool" >&2
    return 1
}

find_game_window() {
    xdotool search --onlyvisible --name "$WINDOW_NAME" 2>/dev/null | head -n 1 || true
}

focus_window() {
    local wid="$1"
    xdotool windowactivate --sync "$wid" >/dev/null 2>&1 || xdotool windowfocus "$wid" >/dev/null 2>&1 || true
    sleep "$FOCUS_SETTLE_DELAY"
}

send_key() {
    local key="$1"
    local delay="$2"
    local wid=""
    wid="$(find_game_window)"
    if [[ -n "$wid" ]]; then
        focus_window "$wid"
        xdotool key --clearmodifiers --window "$wid" "$key"
    else
        xdotool key --clearmodifiers "$key"
    fi
    sleep "$delay"
}

build_release() {
    status "building ZLUDA release"
    (
        cd "$ZLUDA_DIR"
        LIBRARY_PATH=/tmp/stub_libs:ext/rocm/lib cargo build --release -p zluda
    )
}

deploy_binaries() {
    status "deploying libnvcuda.so"
    install_if_changed 755 "$LIBCUDA_SRC" "$PROTON_LIB64_DIR/libcuda.so"
    install_if_changed 755 "$LIBCUDA_SRC" "$PROTON_LIB64_DIR/libcuda.so.1"
    install_if_changed 755 "$LIBCUDA_SRC" "$WINE_UNIX_DIR/libcuda.so"
    install_if_changed 755 "$LIBCUDA_SRC" "$WINE_UNIX_DIR/libcuda.so.1"
}

deploy_protonfixes() {
    status "deploying Batman protonfixes"
    install_if_changed 644 "$PROTONFIX_SRC" "$PROTONFIX_DST"
    install_if_changed 644 "$PROTONFIX_SRC" "$PROTONFIX_PKG_DST"
    find "$COMPAT_TOOL_DIR/protonfixes" -type d -name '__pycache__' -prune -exec rm -rf {} +
}

clear_caches() {
    status "clearing ZLUDA compute cache"
    rm -f "$COMPUTE_CACHE_DB"
}

launch_game() {
    status "launching app ${APP_ID} via Steam CLI"
    "$STEAM_CLI" -applaunch "$APP_ID" >/tmp/batman_steam_launch.log 2>&1 &
    local steam_pid=$!
    status "launch requested (steam pid ${steam_pid}); log: /tmp/batman_steam_launch.log"
}

wait_for_game_process() {
    local log_start="$1"
    local attempts=$((GAME_TIMEOUT / 2))
    status "waiting for BatmanAK.exe"
    for ((i = 1; i <= attempts; i++)); do
        if game_running; then
            status "BatmanAK.exe detected"
            return 0
        fi
        local crash_line=""
        crash_line="$(latest_new_log_match "$log_start" '\\*\\*\\* SIGABRT|GPU fault likely|\\*\\*\\* SIGSEGV' || true)"
        if [[ -n "$crash_line" ]]; then
            echo "Error: crash detected before BatmanAK.exe became stable: $crash_line" >&2
            return 1
        fi
        sleep 2
    done
    echo "Error: timed out waiting for BatmanAK.exe" >&2
    return 1
}

wait_for_menu_ready() {
    local log_start="$1"
    local attempts=$((MENU_TIMEOUT / 2))
    local window_seen_at=""
    status "waiting for menu-ready state"
    for ((i = 1; i <= attempts; i++)); do
        local crash_line=""
        crash_line="$(latest_new_log_match "$log_start" '\\*\\*\\* SIGABRT|GPU fault likely|\\*\\*\\* SIGSEGV' || true)"
        if [[ -n "$crash_line" ]]; then
            echo "Error: crash detected before menu-ready state: $crash_line" >&2
            return 1
        fi
        if ! game_running; then
            echo "Error: BatmanAK.exe exited before menu-ready state" >&2
            return 1
        fi
        local kernels=""
        local idle_calls=""
        local wid=""
        local visible_for=0
        kernels="$(count_new_log_matches "$log_start" 'cuLaunchKernel')"
        idle_calls="$(count_new_log_matches "$log_start" 'cuEventRecord|cuEventSynchronize|cuCtxGetCurrent')"
        wid="$(find_game_window)"
        if [[ -n "$wid" ]]; then
            if [[ -z "$window_seen_at" ]]; then
                window_seen_at="$SECONDS"
            fi
            visible_for=$((SECONDS - window_seen_at))
        else
            window_seen_at=""
        fi
        if [[ "$kernels" -eq 0 && "$idle_calls" -ge "$MENU_IDLE_CALL_THRESHOLD" && -n "$wid" && "$visible_for" -ge "$MENU_VISIBLE_STABLE_SECONDS" ]]; then
            status "menu ready (idle_calls=${idle_calls}, kernels=${kernels}, window=${wid}, visible_for=${visible_for}s)"
            return 0
        fi
        if (( i % 10 == 0 )); then
            status "menu wait: idle_calls=${idle_calls} kernels=${kernels} window=${wid:-none} visible_for=${visible_for}s"
        fi
        sleep 2
    done
    echo "Error: timed out waiting for Batman menu-ready state" >&2
    return 1
}

run_benchmark_sequence() {
    send_key Return 2
    send_key Return 10
    send_key Up 2
    send_key Up 2
    send_key Return 2
    send_key Up 2
    send_key Return 0
}

start_benchmark() {
    status "starting benchmark sequence"
    status "waiting ${INITIAL_MENU_SETTLE}s initial settle + ${INITIAL_KEYPRESS_EXTRA_DELAY}s extra pre-key delay"
    sleep "$INITIAL_MENU_SETTLE"
    sleep "$INITIAL_KEYPRESS_EXTRA_DELAY"
    run_benchmark_sequence
}

retry_benchmark_sequence() {
    status "benchmark still idle; retrying benchmark sequence once"
    run_benchmark_sequence
}

wait_for_benchmark_start() {
    local log_start="$1"
    local attempts=$((BENCHMARK_START_TIMEOUT / 2))
    local retry_after_attempt=$((BENCHMARK_RETRY_AFTER_SECONDS / 2))
    local retried=0
    status "waiting for benchmark kernels"
    for ((i = 1; i <= attempts; i++)); do
        local crash_line=""
        crash_line="$(latest_new_log_match "$log_start" '\\*\\*\\* SIGABRT|GPU fault likely|\\*\\*\\* SIGSEGV' || true)"
        if [[ -n "$crash_line" ]]; then
            echo "Error: crash detected before benchmark start: $crash_line" >&2
            return 1
        fi
        if ! game_running; then
            echo "Error: BatmanAK.exe exited before benchmark kernels appeared" >&2
            return 1
        fi
        local kernels=""
        kernels="$(count_new_log_matches "$log_start" 'cuLaunchKernel')"
        if [[ "$kernels" -ge 50 ]]; then
            status "benchmark started (cuLaunchKernel=${kernels})"
            return 0
        fi
        if [[ "$retried" -eq 0 && "$kernels" -eq 0 && "$i" -ge "$retry_after_attempt" ]]; then
            retried=1
            retry_benchmark_sequence
        fi
        if (( i % 10 == 0 )); then
            status "benchmark wait: cuLaunchKernel=${kernels}"
        fi
        sleep 2
    done
    echo "Error: timed out waiting for benchmark kernels" >&2
    return 1
}

monitor_benchmark_outcome() {
    local log_start="$1"
    local attempts=$((BENCHMARK_MONITOR_TIMEOUT / 5))
    status "monitoring benchmark outcome"
    for ((i = 1; i <= attempts; i++)); do
        local crash_line=""
        crash_line="$(latest_new_log_match "$log_start" '\\*\\*\\* SIGABRT|GPU fault likely|\\*\\*\\* SIGSEGV' || true)"
        if [[ -n "$crash_line" ]]; then
            status "crash detected: ${crash_line}"
            return 2
        fi
        if ! game_running; then
            status "BatmanAK.exe exited without an explicit crash marker"
            return 0
        fi
        if (( i % 12 == 0 )); then
            local kernels=""
            kernels="$(count_new_log_matches "$log_start" 'cuLaunchKernel')"
            status "benchmark still running (cuLaunchKernel=${kernels})"
        fi
        sleep 5
    done
    status "monitor timeout reached; benchmark still appears to be running"
}

latest_capture_with_prefix() {
    local prefix="$1"
    local newest=""
    local newest_mtime=0
    local path=""
    shopt -s nullglob
    for path in /tmp/"${prefix}"*.bin; do
        [[ -f "$path" ]] || continue
        local mtime=0
        mtime="$(stat -c %Y "$path" 2>/dev/null || echo 0)"
        if (( mtime >= newest_mtime )); then
            newest_mtime="$mtime"
            newest="$path"
        fi
    done
    shopt -u nullglob
    printf '%s\n' "$newest"
}

describe_path() {
    local label="$1"
    local path="$2"
    if [[ -n "$path" && -e "$path" ]]; then
        status "$label: $(stat -c '%y %n' "$path")"
    else
        status "$label: missing"
    fi
}

RESOLVED_GOOD_CAPTURE=""
RESOLVED_BAD_CAPTURE=""

resolve_replay_captures() {
    RESOLVED_GOOD_CAPTURE="$GOOD_CAPTURE_PATH"
    RESOLVED_BAD_CAPTURE="$BAD_CAPTURE_PATH"
    if [[ -z "$RESOLVED_GOOD_CAPTURE" ]]; then
        RESOLVED_GOOD_CAPTURE="$(latest_capture_with_prefix "zluda_sort_replay_capture_good_bit")"
    fi
    if [[ -z "$RESOLVED_BAD_CAPTURE" ]]; then
        RESOLVED_BAD_CAPTURE="$(latest_capture_with_prefix "zluda_sort_replay_capture_bit")"
    fi
}

log_replay_capture_status() {
    describe_path "deployed lib" "$PROTON_LIB64_DIR/libcuda.so"
    describe_path "log file" "$LOG_FILE"
    describe_path "good capture" "$RESOLVED_GOOD_CAPTURE"
    describe_path "bad capture" "$RESOLVED_BAD_CAPTURE"

    if [[ -z "$RESOLVED_GOOD_CAPTURE" || ! -f "$RESOLVED_GOOD_CAPTURE" ]]; then
        if [[ -f "$PROTON_LIB64_DIR/libcuda.so" && -f "$LOG_FILE" ]] && \
           [[ "$(stat -c %Y "$LOG_FILE")" -lt "$(stat -c %Y "$PROTON_LIB64_DIR/libcuda.so")" ]]; then
            status "no good capture yet: deployed runtime is newer than the current log, so Batman has not been run since the last deploy"
        else
            status "no good capture yet: run Batman once with the deployed runtime to produce /tmp/zluda_sort_replay_capture_good_bit*_pass*.bin"
        fi
        return 1
    fi
    if [[ -z "$RESOLVED_BAD_CAPTURE" || ! -f "$RESOLVED_BAD_CAPTURE" ]]; then
        status "no bad capture available yet: expected /tmp/zluda_sort_replay_capture_bit*_pass*.bin"
        return 1
    fi
    return 0
}

compare_replay_captures() {
    resolve_replay_captures
    if ! log_replay_capture_status; then
        return 0
    fi
    status "running offline replay capture diff"
    (
        cd "$ZLUDA_DIR"
        ZLUDA_SORT_REPLAY_GOOD_CAPTURE="$RESOLVED_GOOD_CAPTURE" \
        ZLUDA_SORT_REPLAY_BAD_CAPTURE="$RESOLVED_BAD_CAPTURE" \
        LD_LIBRARY_PATH="$(pwd)/ext/rocm/lib" \
        LIBRARY_PATH=/tmp/stub_libs:ext/rocm/lib \
        cargo test -p ptx sort_replay_compare_good_and_bad_captures_live -- --ignored --nocapture
    )
}

watch_replay_captures() {
    local start_ts="$SECONDS"
    local deadline=$((start_ts + WATCH_TIMEOUT))
    local next_report="$start_ts"
    local waiting_note=""
    status "watching for good replay capture (timeout=${WATCH_TIMEOUT}s interval=${WATCH_INTERVAL}s)"
    while (( SECONDS < deadline )); do
        resolve_replay_captures
        if [[ -n "$RESOLVED_GOOD_CAPTURE" && -f "$RESOLVED_GOOD_CAPTURE" ]]; then
            status "detected good replay capture; running offline compare"
            compare_replay_captures
            return $?
        fi
        if [[ -f "$PROTON_LIB64_DIR/libcuda.so" && -f "$LOG_FILE" ]] && \
           [[ "$(stat -c %Y "$LOG_FILE")" -lt "$(stat -c %Y "$PROTON_LIB64_DIR/libcuda.so")" ]]; then
            waiting_note="waiting for the first post-deploy Batman run"
        else
            waiting_note="waiting for /tmp/zluda_sort_replay_capture_good_bit*_pass*.bin"
        fi
        if (( SECONDS >= next_report )); then
            status "$waiting_note"
            describe_path "deployed lib" "$PROTON_LIB64_DIR/libcuda.so"
            describe_path "log file" "$LOG_FILE"
            describe_path "good capture" "$RESOLVED_GOOD_CAPTURE"
            describe_path "bad capture" "$RESOLVED_BAD_CAPTURE"
            next_report=$((SECONDS + 60))
        fi
        sleep "$WATCH_INTERVAL"
    done
    echo "Error: timed out waiting for a good replay capture after ${WATCH_TIMEOUT}s" >&2
    return 1
}

run_benchmark() {
    require_cmd xdotool
    discover_x_session
    local log_start="$1"
    wait_for_game_process "$log_start"
    wait_for_menu_ready "$log_start"
    local benchmark_log_start=""
    benchmark_log_start="$(log_cursor)"
    start_benchmark
    wait_for_benchmark_start "$benchmark_log_start"
    monitor_benchmark_outcome "$benchmark_log_start" || return $?
}

require_dir "$REPO_DIR"
require_dir "$ZLUDA_DIR"
require_dir "$COMPAT_TOOL_DIR"
require_file "$PROTONFIX_SRC"

COMPARE_ONLY=0
if [[ "$DO_COMPARE_CAPTURES" -eq 1 && "$DO_BUILD" -eq 0 && "$DO_LAUNCH" -eq 0 ]]; then
    COMPARE_ONLY=1
fi

if [[ "$DO_BUILD" -eq 1 ]]; then
    build_release
fi

if [[ "$COMPARE_ONLY" -eq 0 ]]; then
    require_file "$LIBCUDA_SRC"
    deploy_binaries
    deploy_protonfixes
    clear_caches
else
    status "compare-only mode: skipping deploy and cache clear"
fi

if [[ "$DO_LAUNCH" -eq 1 ]]; then
    require_cmd "$STEAM_CLI"
    LOG_START="$(log_cursor)"
    launch_game
    if [[ "$DO_BENCHMARK" -eq 1 ]]; then
        run_benchmark "$LOG_START"
    fi
fi

if [[ "$DO_COMPARE_CAPTURES" -eq 1 ]]; then
    if [[ "$DO_WATCH_CAPTURES" -eq 1 ]]; then
        watch_replay_captures
    else
        compare_replay_captures
    fi
fi

status "done"
