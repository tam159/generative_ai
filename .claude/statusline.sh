#!/bin/bash
input=$(cat)

MODEL=$(echo "$input" | jq -r '.model.display_name')
CTX_SIZE=$(echo "$input" | jq -r '.context_window.context_window_size // 0')
DIR=$(echo "$input" | jq -r '.workspace.current_dir')
COST=$(echo "$input" | jq -r '.cost.total_cost_usd // 0')
PCT=$(echo "$input" | jq -r '.context_window.used_percentage // 0' | cut -d. -f1)
DURATION_MS=$(echo "$input" | jq -r '.cost.total_duration_ms // 0')
LINES_ADDED=$(echo "$input" | jq -r '.cost.total_lines_added // 0')
LINES_REMOVED=$(echo "$input" | jq -r '.cost.total_lines_removed // 0')
SESSION_NAME=$(echo "$input" | jq -r '.session_name // empty')

# Format context window size
if [ "$CTX_SIZE" -ge 1000000 ] 2>/dev/null; then
    CTX_LABEL="$(echo "$CTX_SIZE" | awk '{printf "%gM", $1/1000000}') context"
elif [ "$CTX_SIZE" -ge 1000 ] 2>/dev/null; then
    CTX_LABEL="$(echo "$CTX_SIZE" | awk '{printf "%gK", $1/1000}') context"
else
    CTX_LABEL=""
fi
if [ -n "$CTX_LABEL" ] && [[ "$MODEL" != *"$CTX_LABEL"* ]]; then
    MODEL_DISPLAY="$MODEL ($CTX_LABEL)"
else
    MODEL_DISPLAY="$MODEL"
fi

CYAN='\033[36m'; GREEN='\033[32m'; YELLOW='\033[33m'; RED='\033[31m'; RESET='\033[0m'

# Pick bar color based on context usage
if [ "$PCT" -ge 90 ]; then BAR_COLOR="$RED"
elif [ "$PCT" -ge 70 ]; then BAR_COLOR="$YELLOW"
else BAR_COLOR="$GREEN"; fi

FILLED=$((PCT / 10)); EMPTY=$((10 - FILLED))
printf -v FILL "%${FILLED}s"; printf -v PAD "%${EMPTY}s"
BAR="${FILL// /█}${PAD// /░}"

MINS=$((DURATION_MS / 60000)); SECS=$(((DURATION_MS % 60000) / 1000))

BRANCH=""
git -C "$DIR" rev-parse --git-dir > /dev/null 2>&1 && BRANCH=" | 🌿 $(git -C "$DIR" branch --show-current 2>/dev/null)"

FIVE_H=$(echo "$input" | jq -r '.rate_limits.five_hour.used_percentage // empty')
WEEK=$(echo "$input" | jq -r '.rate_limits.seven_day.used_percentage // empty')
LIMITS=""
[ -n "$FIVE_H" ] && LIMITS="5h:$(printf '%.0f' "$FIVE_H")%"
[ -n "$WEEK" ] && LIMITS="${LIMITS:+$LIMITS }7d:$(printf '%.0f' "$WEEK")%"
LIMITS_PART=""
[ -n "$LIMITS" ] && LIMITS_PART=" | ${YELLOW}${LIMITS}${RESET}"

# Build clickable GitHub repo link if remote exists
REMOTE=$(git -C "$DIR" remote get-url origin 2>/dev/null | sed 's/git@github.com:/https:\/\/github.com\//' | sed 's/\.git$//')
COST_FMT=$(printf '$%.2f' "$COST")
LINES_PART=" | +${LINES_ADDED}/-${LINES_REMOVED}"
SESSION_PART=""
[ -n "$SESSION_NAME" ] && SESSION_PART=" | ${CYAN}${SESSION_NAME}${RESET}"
SUFFIX=" | ${BAR_COLOR}${BAR}${RESET} ${PCT}%${LIMITS_PART} | ${YELLOW}${COST_FMT}${RESET}${LINES_PART} | ⏱️ ${MINS}m ${SECS}s"
if [ -n "$REMOTE" ]; then
    REPO_NAME=$(basename "$REMOTE")
    REPO_LINK=$(printf '%b' "\e]8;;${REMOTE}\a${REPO_NAME}\e]8;;\a")
    printf '%b\n' "${CYAN}[$MODEL_DISPLAY]${RESET} 🔗 ${REPO_LINK}${BRANCH}${SUFFIX}"
else
    printf '%b\n' "${CYAN}[$MODEL_DISPLAY]${RESET} 📁 ${DIR##*/}${BRANCH}${SUFFIX}"
fi
