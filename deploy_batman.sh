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
  --no-clean-dumps     Skip clearing Batman crash/GPU dump artifacts before launch
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
TITLE_PROMPT_KEY="${BATMAN_TITLE_PROMPT_KEY:-Return}"
TITLE_PROMPT_RETRY_KEY="${BATMAN_TITLE_PROMPT_RETRY_KEY:-Return}"
ATTRACT_ESCAPE_KEY="${BATMAN_ATTRACT_ESCAPE_KEY:-Escape}"
GAME_PROCESS_PATTERN="BatmanAK.exe -nologo"
DO_BUILD=1
DO_LAUNCH=1
DO_BENCHMARK=1
DO_CLEAN_DUMPS=1
DO_COMPARE_CAPTURES=0
DO_WATCH_CAPTURES=0
GOOD_CAPTURE_PATH=""
BAD_CAPTURE_PATH=""
WATCH_TIMEOUT=28800
WATCH_INTERVAL=10

GAME_TIMEOUT=240
BENCHMARK_START_TIMEOUT=360
BENCHMARK_MONITOR_TIMEOUT=900
INITIAL_MENU_SETTLE=30
TITLE_PROMPT_ADVANCE_DELAY=2
SAVE_SLOT_SELECT_DELAY=10
OPTIONS_MENU_SETTLE_DELAY=2
BENCHMARK_MENU_SETTLE_DELAY="${BATMAN_BENCHMARK_MENU_SETTLE_DELAY:-0}"
FOCUS_SETTLE_DELAY=0.2
# The benchmark can sit on a long loading screen before the first CUDA launches
# appear in the log. Retrying too early just kicks the UI again while the load
# is still in progress, so keep the default retry comfortably past that handoff.
BENCHMARK_RETRY_AFTER_SECONDS="${BATMAN_BENCHMARK_RETRY_AFTER_SECONDS:-180}"

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
        --no-clean-dumps)
            DO_CLEAN_DUMPS=0
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
PROTONFIX_ENV_DST="${PROTONFIX_DST%.py}.env"
PROTONFIX_PKG_ENV_DST="${PROTONFIX_PKG_DST%.py}.env"
COMPUTE_CACHE_DB="${HOME}/.cache/zluda/ComputeCache/zluda.db"
ZLUDA_CARGO_SCRIPT="$REPO_DIR/scripts/zluda-cargo.sh"
BATMAN_DUMP_CLEANUP_SCRIPT="$REPO_DIR/scripts/cleanup-batman-dumps.sh"
BATMAN_PROTONFIX_OVERRIDE_VARS=(
    ZLUDA_LAUNCH_MUTEX
    ZLUDA_LOG_ALL
    ZLUDA_LOG_LAUNCHES
    ZLUDA_VALIDATE_SORT
    ZLUDA_VALIDATE_SORT_SAFE
    ZLUDA_TRACE_SORT_COPY
    ZLUDA_CREATEGRID_TOUCH_LOG
    ZLUDA_CREATEGRID_SNAPSHOT_LOG
    ZLUDA_CREATECELL_SNAPSHOT_LOG
    ZLUDA_SORT_REPLAY_CAPTURE_BIT
    ZLUDA_MS4_CAPTURE
    ZLUDA_TFS_CAPTURE
    ZLUDA_IOFX_CAPTURE
    ZLUDA_IOFX_BBOX_GUARD
    ZLUDA_IOFX_BBOX_CAPTURE
    ZLUDA_DOWNSWEEP_CAPTURE
    ZLUDA_COUNTER_SNAPSHOT
    ZLUDA_SORT_ZERO_INPUT_SLACK
    ZLUDA_NULL_STREAM_STRICT
    ZLUDA_PHYSX_WAR_BARRIERS
    ZLUDA_DTOH_LOG
    ZLUDA_KARG_LOG
    ZLUDA_HOSTBUF_SAMPLE
    PROTON_LOG
    DXVK_LOG_LEVEL
    WINEDEBUG
    AMD_LOG_LEVEL
)

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

write_batman_protonfix_overrides() {
    local dst="$1"
    local tmp
    local wrote=0
    tmp="$(mktemp)"
    : > "$tmp"
    for var in "${BATMAN_PROTONFIX_OVERRIDE_VARS[@]}"; do
        if [[ ${!var+x} ]]; then
            printf '%s=%s\n' "$var" "${!var}" >> "$tmp"
            wrote=1
        fi
    done
    if (( wrote )); then
        install -D -m 644 "$tmp" "$dst"
    else
        rm -f "$dst"
    fi
    rm -f "$tmp"
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

list_game_pids() {
    ps -eo pid=,cmd= | grep -F "$GAME_PROCESS_PATTERN" | grep -v grep | awk '{print $1}'
}

terminate_game_pid() {
    local pid="$1"
    local attempts=0
    if ! kill -0 "$pid" >/dev/null 2>&1; then
        return 0
    fi
    kill "$pid" >/dev/null 2>&1 || true
    for ((attempts = 1; attempts <= 10; attempts++)); do
        if ! kill -0 "$pid" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done
    status "BatmanAK.exe pid ${pid} ignored TERM; sending KILL"
    kill -9 "$pid" >/dev/null 2>&1 || true
    for ((attempts = 1; attempts <= 5; attempts++)); do
        if ! kill -0 "$pid" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done
    echo "Error: failed to stop stale BatmanAK.exe pid ${pid}" >&2
    return 1
}

clear_stale_game_processes() {
    local pids=()
    local pid=""
    while read -r pid; do
        [[ -n "$pid" ]] || continue
        pids+=("$pid")
    done < <(list_game_pids)
    if [[ "${#pids[@]}" -eq 0 ]]; then
        return 0
    fi
    status "stopping stale BatmanAK.exe pids: ${pids[*]}"
    for pid in "${pids[@]}"; do
        terminate_game_pid "$pid"
    done
}

discover_x_session() {
    usable_xauthority_file() {
        local auth="$1"
        [[ -n "$auth" && -f "$auth" && -r "$auth" ]]
    }

    x_display_socket_ready() {
        local disp="$1"
        local display_num="${disp#:}"
        [[ "$display_num" =~ ^[0-9]+$ ]] || return 1
        [[ -S "/tmp/.X11-unix/X${display_num}" ]]
    }

    probe_x_session() {
        local disp="$1"
        local auth="${2-}"
        x_display_socket_ready "$disp" || return 1
        if ! command -v xdotool >/dev/null 2>&1; then
            return 0
        fi
        if usable_xauthority_file "$auth"; then
            DISPLAY="$disp" XAUTHORITY="$auth" xdotool getdisplaygeometry >/dev/null 2>&1
        else
            env -u XAUTHORITY DISPLAY="$disp" xdotool getdisplaygeometry >/dev/null 2>&1
        fi
    }

    use_x_session() {
        local disp="$1"
        local auth="${2-}"
        local source="$3"
        X_DISPLAY="$disp"
        export DISPLAY="$X_DISPLAY"
        if usable_xauthority_file "$auth"; then
            X_AUTHORITY="$auth"
            export XAUTHORITY="$X_AUTHORITY"
            status "${source} DISPLAY=$DISPLAY XAUTHORITY=$XAUTHORITY"
        else
            X_AUTHORITY=""
            unset XAUTHORITY || true
            status "${source} DISPLAY=$DISPLAY (no readable XAUTHORITY)"
        fi
        return 0
    }

    local pid=""
    local disp=""
    local auth=""
    if [[ -n "$X_DISPLAY" ]] && probe_x_session "$X_DISPLAY" "$X_AUTHORITY"; then
        use_x_session "$X_DISPLAY" "$X_AUTHORITY" "using requested X session"
        return 0
    fi
    while read -r pid _; do
        [[ -r "/proc/$pid/environ" ]] || continue
        disp="$(tr '\0' '\n' < "/proc/$pid/environ" | sed -n 's/^DISPLAY=//p' | head -n 1)"
        auth="$(tr '\0' '\n' < "/proc/$pid/environ" | sed -n 's/^XAUTHORITY=//p' | head -n 1)"
        if [[ -n "$disp" ]] && probe_x_session "$disp" "$auth"; then
            use_x_session "$disp" "$auth" "using discovered X session"
            return 0
        fi
    done < <(ps -eo pid,cmd | grep -E 'kwin_x11|plasmashell|startplasma-x11|gamescope|Xwayland|Xorg' | grep -v grep)

    shopt -s nullglob
    local fallback_auth=(/run/user/"$(id -u)"/xauth_* "$HOME/.Xauthority")
    local fallback_socket=""
    shopt -u nullglob
    for fallback_socket in /tmp/.X11-unix/X*; do
        [[ -S "$fallback_socket" ]] || continue
        disp=":${fallback_socket##*X}"
        for auth in "${fallback_auth[@]}"; do
            if probe_x_session "$disp" "$auth"; then
                use_x_session "$disp" "$auth" "using fallback X session"
                return 0
            fi
        done
        if probe_x_session "$disp" ""; then
            use_x_session "$disp" "" "using fallback X session"
            return 0
        fi
    done
    echo "Error: could not discover a usable X session for xdotool" >&2
    return 1
}

find_game_window() {
    local wid=""
    local best_wid=""
    local geom=""
    local width=""
    local height=""
    local area=0
    local best_area=-1
    while read -r wid; do
        [[ -n "$wid" ]] || continue
        geom="$(xdotool getwindowgeometry --shell "$wid" 2>/dev/null || true)"
        width="$(awk -F= '/^WIDTH=/{print $2}' <<<"$geom")"
        height="$(awk -F= '/^HEIGHT=/{print $2}' <<<"$geom")"
        if [[ "$width" =~ ^[0-9]+$ && "$height" =~ ^[0-9]+$ ]]; then
            area=$((width * height))
            if (( area > best_area )); then
                best_area=$area
                best_wid="$wid"
            fi
        fi
    done < <(xdotool search --onlyvisible --name "$WINDOW_NAME" 2>/dev/null || true)
    if [[ -n "$best_wid" ]]; then
        printf '%s\n' "$best_wid"
    fi
}

focus_window() {
    local wid="$1"
    xdotool windowactivate --sync "$wid" >/dev/null 2>&1 || xdotool windowfocus "$wid" >/dev/null 2>&1 || true
    sleep "$FOCUS_SETTLE_DELAY"
}

wait_for_game_window() {
    local timeout="${1:-60}"
    local attempts=$((timeout / 2))
    local wid=""
    local last_wid=""
    local stable_count=0
    for ((i = 1; i <= attempts; i++)); do
        wid="$(find_game_window)"
        if [[ -n "$wid" ]]; then
            if [[ "$wid" == "$last_wid" ]]; then
                stable_count=$((stable_count + 1))
            else
                last_wid="$wid"
                stable_count=1
            fi
            if (( stable_count >= 2 )); then
                focus_window "$wid"
                return 0
            fi
        else
            last_wid=""
            stable_count=0
        fi
        sleep 2
    done
    return 1
}

send_key() {
    local key="$1"
    local delay="$2"
    local active_name=""
    active_name="$(xdotool getactivewindow getwindowname 2>/dev/null || true)"
    if [[ "$active_name" != *"$WINDOW_NAME"* ]]; then
        local wid=""
        if ! wait_for_game_window 10; then
            echo "Error: Batman window not visible before sending key '$key'" >&2
            return 1
        fi
        wid="$(find_game_window)"
        if [[ -z "$wid" ]]; then
            echo "Error: Batman window disappeared before sending key '$key'" >&2
            return 1
        fi
        focus_window "$wid"
    fi
    xdotool key --clearmodifiers "$key"
    sleep "$delay"
}

build_release() {
    status "building ZLUDA release"
    bash "$ZLUDA_CARGO_SCRIPT" build --release -p zluda
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
    write_batman_protonfix_overrides "$PROTONFIX_ENV_DST"
    write_batman_protonfix_overrides "$PROTONFIX_PKG_ENV_DST"
    find "$COMPAT_TOOL_DIR/protonfixes" -type d -name '__pycache__' -prune -exec rm -rf {} +
}

clear_caches() {
    status "clearing ZLUDA compute cache"
    rm -f "$COMPUTE_CACHE_DB"
}

clean_batman_dumps() {
    status "clearing Batman crash/GPU dump artifacts"
    bash "$BATMAN_DUMP_CLEANUP_SCRIPT"
}

launch_game() {
    clear_stale_game_processes
    discover_x_session
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

run_benchmark_menu_sequence() {
    send_key Up 2
    send_key Up 2
    send_key Return 2
    if (( OPTIONS_MENU_SETTLE_DELAY > 0 )); then
        status "waiting ${OPTIONS_MENU_SETTLE_DELAY}s for options menu"
        sleep "$OPTIONS_MENU_SETTLE_DELAY"
    fi
    send_key Up 2
    send_key Return 0
}

run_full_benchmark_sequence() {
    local title_prompt_key="${1:-$TITLE_PROMPT_KEY}"
    status "advancing past title prompt"
    send_key "$title_prompt_key" "$TITLE_PROMPT_ADVANCE_DELAY"
    status "accepting highlighted default profile"
    send_key Return "$SAVE_SLOT_SELECT_DELAY"
    if (( BENCHMARK_MENU_SETTLE_DELAY > 0 )); then
        status "waiting ${BENCHMARK_MENU_SETTLE_DELAY}s for main menu"
        sleep "$BENCHMARK_MENU_SETTLE_DELAY"
    fi
    run_benchmark_menu_sequence
}

start_benchmark() {
    status "starting benchmark sequence"
    if wait_for_game_window 60; then
        status "Batman window detected; waiting ${INITIAL_MENU_SETTLE}s before first input"
    else
        status "Batman window not visible yet; waiting ${INITIAL_MENU_SETTLE}s before first input"
    fi
    sleep "$INITIAL_MENU_SETTLE"
    run_full_benchmark_sequence
}

retry_benchmark_sequence() {
    status "benchmark still idle; retrying full benchmark sequence once"
    if [[ -n "$TITLE_PROMPT_RETRY_KEY" && "$TITLE_PROMPT_RETRY_KEY" != "$TITLE_PROMPT_KEY" ]]; then
        status "retrying title prompt with alternate key ${TITLE_PROMPT_RETRY_KEY}"
        run_full_benchmark_sequence "$TITLE_PROMPT_RETRY_KEY"
    else
        run_full_benchmark_sequence
    fi
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
    # 2026-04-28 Phase 3a: kernel-quiescence detection + auto re-entry test.
    # If we've seen lots of kernels and growth has stalled, assume the
    # benchmark completed. Take a screenshot, send Esc + Enter to re-trigger
    # the benchmark menu (back → re-enter), then keep monitoring. If a
    # second "completion" or a crash happens, we have a 2-cycle data point.
    # Set ZLUDA_AUTO_REENTRY=0 to disable.
    local auto_reentry="${ZLUDA_AUTO_REENTRY:-1}"
    local prev_kernels=0
    local stall_count=0
    local STALL_THRESHOLD=8           # 8 × 5s = 40s of no kernel growth
    local KERNELS_FOR_COMPLETE=200000 # only consider "completed" if we saw lots
    local reentry_done=0
    status "monitoring benchmark outcome"
    for ((i = 1; i <= attempts; i++)); do
        local crash_line=""
        crash_line="$(latest_new_log_match "$log_start" '\\*\\*\\* SIGABRT|GPU fault likely|\\*\\*\\* SIGSEGV' || true)"
        if [[ -n "$crash_line" ]]; then
            local phase="initial"
            (( reentry_done == 1 )) && phase="reentry"
            status "crash detected (${phase}): ${crash_line}"
            return 2
        fi
        if ! game_running; then
            status "BatmanAK.exe exited without an explicit crash marker"
            return 0
        fi

        # Kernel-quiescence + auto re-entry logic.
        # Idle benchmark menu still ticks ~800 kernels/s (MemCopyBatchedAsync +
        # MS_2 alternating). Real mid-benchmark is many thousands per second.
        # We declare "quiescent" if growth in the 5s tick is below a threshold.
        if (( auto_reentry == 1 )) && (( reentry_done == 0 )); then
            local kernels=""
            kernels="$(count_new_log_matches "$log_start" 'cuLaunchKernel')"
            if (( kernels >= KERNELS_FOR_COMPLETE )); then
                local delta=$((kernels - prev_kernels))
                # Idle adds <5000/5s; mid-benchmark adds tens of thousands.
                if (( delta < 5000 )); then
                    stall_count=$((stall_count + 1))
                else
                    stall_count=0
                fi
                if (( stall_count >= STALL_THRESHOLD )); then
                    status "kernel quiescence at cuLaunchKernel=${kernels} for ${STALL_THRESHOLD}×5s — assuming benchmark completed"
                    # We don't have access to take_screenshot here (defined in
                    # run_benchmark.sh); print a marker line that the wrapping
                    # archiver picks up as a "first benchmark completed" event.
                    status "ATTEMPTING RE-ENTRY: sending Escape, then Enter"
                    discover_x_session >/dev/null 2>&1 || true
                    if command -v xdotool >/dev/null 2>&1; then
                        local wid=""
                        wid="$(find_game_window)" || true
                        [[ -n "$wid" ]] && focus_window "$wid" >/dev/null 2>&1 || true
                        xdotool key --clearmodifiers Escape >/dev/null 2>&1 || true
                        sleep 3
                        xdotool key --clearmodifiers Return >/dev/null 2>&1 || true
                        sleep 1
                        xdotool key --clearmodifiers Return >/dev/null 2>&1 || true
                    fi
                    reentry_done=1
                    status "RE-ENTRY SENT — continuing monitor"
                    stall_count=0
                fi
            fi
            prev_kernels=$kernels
        fi

        if (( i % 12 == 0 )); then
            local kernels=""
            kernels="$(count_new_log_matches "$log_start" 'cuLaunchKernel')"
            local phase="initial"
            (( reentry_done == 1 )) && phase="reentry"
            status "benchmark still running [${phase}] (cuLaunchKernel=${kernels})"
        fi
        sleep 5
    done
    status "monitor timeout reached; benchmark still appears to be running (reentry_done=${reentry_done})"
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
    ZLUDA_SORT_REPLAY_GOOD_CAPTURE="$RESOLVED_GOOD_CAPTURE" \
    ZLUDA_SORT_REPLAY_BAD_CAPTURE="$RESOLVED_BAD_CAPTURE" \
    bash "$ZLUDA_CARGO_SCRIPT" test -p ptx --features has_rocm sort_replay_compare_good_and_bad_captures_live -- --ignored --nocapture
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
    local benchmark_log_start=""
    benchmark_log_start="$(log_cursor)"
    start_benchmark "$benchmark_log_start"
    wait_for_benchmark_start "$benchmark_log_start"
    monitor_benchmark_outcome "$benchmark_log_start" || return $?
}

require_dir "$REPO_DIR"
require_dir "$ZLUDA_DIR"
require_dir "$COMPAT_TOOL_DIR"
require_file "$PROTONFIX_SRC"
require_file "$ZLUDA_CARGO_SCRIPT"
require_file "$BATMAN_DUMP_CLEANUP_SCRIPT"

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
    if [[ "$DO_CLEAN_DUMPS" -eq 1 ]]; then
        clean_batman_dumps
    fi
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
