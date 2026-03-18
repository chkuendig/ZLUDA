#!/bin/bash
# Automate Batman: Arkham Knight menu navigation to start the perf test level.
# Usage: ./test_launch_level.sh [screenshot_dir]
#
# Sequence from main menu: Enter (continue) → Enter (confirm) →
# wait for level load → Up Up Enter (perf test) → Up Enter (start)
#
# Prerequisites: xdotool, spectacle (KDE screenshot tool)

SCREENSHOT_DIR="${1:-/tmp/zluda_screenshots}"
mkdir -p "$SCREENSHOT_DIR"

take_screenshot() {
    local name="$1"
    local path="$SCREENSHOT_DIR/${name}_$(date +%s).png"
    # Use spectacle in non-interactive mode
    spectacle -b -n -o "$path" 2>/dev/null && echo "Screenshot: $path" || echo "Screenshot failed"
}

wait_for_game() {
    echo "Waiting for game window..."
    for i in $(seq 1 60); do
        WID=$(xdotool search --name "Batman" 2>/dev/null | head -1)
        if [ -n "$WID" ]; then
            echo "Found game window: $WID"
            xdotool windowfocus "$WID" 2>/dev/null
            return 0
        fi
        sleep 2
    done
    echo "Game window not found after 120s"
    return 1
}

send_key() {
    local key="$1"
    local delay="${2:-1}"
    echo "Sending key: $key"
    xdotool key "$key"
    sleep "$delay"
}

# Wait for game to be ready
wait_for_game || exit 1

echo "Taking pre-menu screenshot..."
take_screenshot "01_menu"

echo "Navigating menu..."
sleep 2
send_key "Return" 3    # Continue / New Game
take_screenshot "02_after_first_enter"
send_key "Return" 3    # Confirm
take_screenshot "03_after_second_enter"

echo "Waiting for level to load (watching ZLUDA log)..."
PREV_LINES=0
for i in $(seq 1 120); do  # Wait up to 4 minutes
    LINES=$(wc -l < /tmp/zluda_tex.log 2>/dev/null || echo 0)
    if [ "$LINES" -gt 500 ]; then
        echo "ZLUDA log has $LINES lines — level is loading/loaded"
        break
    fi
    if [ "$((i % 15))" -eq 0 ]; then
        echo "  Still waiting... ($LINES lines in log)"
        take_screenshot "04_loading_${i}"
    fi
    sleep 2
done

take_screenshot "05_loaded"
echo "Done. Screenshots in $SCREENSHOT_DIR"
echo "ZLUDA log: $(wc -l < /tmp/zluda_tex.log 2>/dev/null) lines"
echo "Launches: $(grep -c 'launch' /tmp/zluda_tex.log 2>/dev/null)"
