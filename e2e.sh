#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REAL_E2E_DIR="$SCRIPT_DIR/real_e2e"
RUNS_DIR="$REAL_E2E_DIR/runs"
CENTRAL_LOG="$RUNS_DIR/e2e_central.log"

RUST_BIN="$SCRIPT_DIR/docling-rs/target/release/docling-rs"
PYTHON_BIN="docling"

OUTPUT_FORMATS=("md")

SUPPORTED_TYPES=(pdf docx pptx xlsx csv html markdown image latex)

SKIP_REVIEW="${SKIP_REVIEW:-}"

# ── Helpers ──────────────────────────────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

log()   { echo -e "${CYAN}[e2e]${NC} $*"; }
ok()    { echo -e "${GREEN}  ✓${NC} $*"; }
warn()  { echo -e "${YELLOW}  ⚠${NC} $*"; }
fail()  { echo -e "${RED}  ✗${NC} $*"; }
fatal() { echo -e "${RED}[e2e] ERROR:${NC} $*" >&2; exit 1; }

usage() {
    cat <<EOF
Usage: $0 <doc_type> <command>

Commands:
  run       Run e2e comparison for the given document type
  list      List available document types and their sample files

Document types: ${SUPPORTED_TYPES[*]}

Examples:
  $0 pdf run         # Run PDF e2e comparison
  $0 docx run        # Run DOCX e2e comparison
  $0 list            # List all types and files
EOF
    exit 1
}

# Map our folder names to the --from values each CLI expects.
# Both CLIs auto-detect from extension, so --from is only needed
# for the "markdown" folder whose files end in .md (CLI value: "md").
rust_from_flag() {
    case "$1" in
        markdown) echo "md" ;;
        *)        echo "" ;;
    esac
}

python_from_flag() {
    case "$1" in
        markdown) echo "md" ;;
        *)        echo "" ;;
    esac
}

# ── Pre-flight checks ───────────────────────────────────────────────────────

check_prerequisites() {
    if [[ ! -x "$RUST_BIN" ]]; then
        log "Rust release binary not found. Building..."
        (cd "$SCRIPT_DIR/docling-rs" && cargo build --release) \
            || fatal "Failed to build Rust binary"
    fi
    ok "Rust binary: $RUST_BIN"

    if ! command -v "$PYTHON_BIN" &>/dev/null; then
        fatal "Python docling CLI not found. Install with: pip install docling"
    fi
    ok "Python binary: $(which $PYTHON_BIN)"

    if ! command -v cursor &>/dev/null; then
        warn "Cursor CLI not found — comparison review will be skipped"
    fi
}

# ── Run conversion for one engine ────────────────────────────────────────────

run_engine() {
    local engine="$1"    # rust | python
    local doc_type="$2"
    local input_dir="$3"
    local output_dir="$4"
    local log_file="$5"

    mkdir -p "$output_dir"

    local from_flag=""
    if [[ "$engine" == "rust" ]]; then
        from_flag=$(rust_from_flag "$doc_type")
    else
        from_flag=$(python_from_flag "$doc_type")
    fi

    local to_args=""
    for fmt in "${OUTPUT_FORMATS[@]}"; do
        to_args+=" --to $fmt"
    done

    local file_count=0
    local success_count=0
    local fail_count=0
    local total_time=0

    for input_file in "$input_dir"/*; do
        [[ -f "$input_file" ]] || continue
        local basename
        basename="$(basename "$input_file")"
        file_count=$((file_count + 1))

        local file_output_dir="$output_dir/$basename"
        mkdir -p "$file_output_dir"

        local cmd=""
        if [[ "$engine" == "rust" ]]; then
            cmd="$RUST_BIN convert"
            [[ -n "$from_flag" ]] && cmd+=" --from $from_flag"
            cmd+=" $to_args --image-export-mode referenced"
            cmd+=" -o $file_output_dir"
            cmd+=" $input_file"
        else
            cmd="$PYTHON_BIN $input_file"
            [[ -n "$from_flag" ]] && cmd+=" --from $from_flag"
            cmd+=" $to_args --image-export-mode referenced"
            cmd+=" --output $file_output_dir"
        fi

        local start_ts end_ts elapsed exit_code
        start_ts=$(date +%s%N 2>/dev/null || python3 -c 'import time; print(int(time.time()*1e9))')

        echo ">>> [$engine] $basename" >> "$log_file"
        echo "    cmd: $cmd" >> "$log_file"

        if eval "$cmd" >> "$log_file" 2>&1; then
            exit_code=0
            success_count=$((success_count + 1))
        else
            exit_code=$?
            fail_count=$((fail_count + 1))
        fi

        end_ts=$(date +%s%N 2>/dev/null || python3 -c 'import time; print(int(time.time()*1e9))')
        elapsed=$(( (end_ts - start_ts) / 1000000 ))
        total_time=$((total_time + elapsed))

        if [[ $exit_code -eq 0 ]]; then
            ok "$basename  (${elapsed}ms)"
        else
            fail "$basename  (exit $exit_code, ${elapsed}ms)"
        fi

        echo "    exit_code: $exit_code  elapsed: ${elapsed}ms" >> "$log_file"
        echo "" >> "$log_file"
    done

    echo ""
    log "$engine summary: $success_count/$file_count succeeded, $fail_count failed, total ${total_time}ms"
    echo ""

    cat >> "$log_file" <<EOF

=== $engine SUMMARY ===
files:     $file_count
succeeded: $success_count
failed:    $fail_count
total_ms:  $total_time
EOF
}

# ── Generate comparison report ───────────────────────────────────────────────

generate_report() {
    local run_dir="$1"
    local doc_type="$2"
    local rust_dir="$run_dir/rust"
    local python_dir="$run_dir/python"
    local report="$run_dir/comparison_report.md"

    cat > "$report" <<EOF
# E2E Comparison Report: $doc_type
**Run:** $(basename "$run_dir")
**Date:** $(date -u +"%Y-%m-%d %H:%M:%S UTC")

## Files Tested
EOF

    local input_dir="$REAL_E2E_DIR/$doc_type"
    for input_file in "$input_dir"/*; do
        [[ -f "$input_file" ]] || continue
        local basename
        basename="$(basename "$input_file")"
        local stem="${basename%.*}"

        echo "" >> "$report"
        echo "### $basename" >> "$report"
        echo "" >> "$report"
        echo "| Format | Rust | Python | Diff |" >> "$report"
        echo "|--------|------|--------|------|" >> "$report"

        for fmt in "${OUTPUT_FORMATS[@]}"; do
            local ext="$fmt"
            local rust_file="$rust_dir/$basename/$stem.$ext"
            local python_file="$python_dir/$basename/$stem.$ext"

            local rust_status="missing"
            local python_status="missing"
            local diff_status="N/A"

            if [[ -f "$rust_file" ]]; then
                local rust_size
                rust_size=$(wc -c < "$rust_file" | tr -d ' ')
                rust_status="${rust_size} bytes"
            fi

            if [[ -f "$python_file" ]]; then
                local python_size
                python_size=$(wc -c < "$python_file" | tr -d ' ')
                python_status="${python_size} bytes"
            fi

            if [[ -f "$rust_file" && -f "$python_file" ]]; then
                if diff -q "$rust_file" "$python_file" &>/dev/null; then
                    diff_status="identical"
                else
                    local diff_lines
                    local diff_raw
                    diff_raw=$(diff "$rust_file" "$python_file" 2>/dev/null || true)
                    diff_lines=$(echo "$diff_raw" | wc -l | tr -d ' ')
                    diff_status="${diff_lines} diff lines"

                    local diff_out="$run_dir/diffs/${basename}.${ext}.diff"
                    mkdir -p "$run_dir/diffs"
                    diff -u "$rust_file" "$python_file" > "$diff_out" 2>/dev/null || true
                fi
            fi

            echo "| $ext | $rust_status | $python_status | $diff_status |" >> "$report"
        done
    done

    cat >> "$report" <<EOF

## Logs
- Rust log: [rust.log](rust.log)
- Python log: [python.log](python.log)
- Diffs: [diffs/](diffs/)
EOF

    ok "Report: $report"
}

# ── Cursor agent review ─────────────────────────────────────────────────────

launch_review() {
    local run_dir="$1"
    local doc_type="$2"

    if ! command -v cursor &>/dev/null; then
        warn "Skipping Cursor agent review (cursor CLI not installed)"
        return 0
    fi

    log "Launching Cursor agent to review results..."

    local prompt
    prompt=$(cat <<PROMPT
You are reviewing the results of an e2e comparison between the Python docling library and its Rust rewrite (docling-rs) for the "$doc_type" document type.

The run directory is: $run_dir

Structure:
- rust/       — output from the Rust binary (docling-rs)
- python/     — output from the Python binary (docling)
- diffs/      — unified diffs between rust and python output for each file
- rust.log    — full Rust conversion log
- python.log  — full Python conversion log
- comparison_report.md — summary of file sizes and diff stats

Your tasks:
1. Read comparison_report.md for an overview
2. Examine the diffs/ folder to understand the differences
3. For each document, compare the Markdown (.md) and JSON (.json) outputs between rust and python
4. Identify gaps in the Rust implementation — things Python handles but Rust does not
5. Categorize issues by severity: CRITICAL (missing content/structure), MODERATE (formatting/ordering differences), MINOR (whitespace/cosmetic)
6. Check the log files for any errors or warnings
7. Write a detailed review to: $run_dir/review.md

Focus on functional correctness gaps that indicate missing or broken features in the Rust rewrite.
PROMPT
)

    cursor agent \
        --print \
        --yolo \
        --trust \
        --workspace "$SCRIPT_DIR" \
        "$prompt" \
        2>&1 | tee "$run_dir/cursor_agent.log"

    if [[ -f "$run_dir/review.md" ]]; then
        ok "Agent review written to: $run_dir/review.md"
    else
        warn "Agent did not produce review.md — check $run_dir/cursor_agent.log"
    fi
}

# ── Central log ──────────────────────────────────────────────────────────────

append_central_log() {
    local run_dir="$1"
    local doc_type="$2"
    local timestamp="$3"

    mkdir -p "$(dirname "$CENTRAL_LOG")"

    local rust_ok=0 rust_fail=0 python_ok=0 python_fail=0

    if [[ -f "$run_dir/rust.log" ]]; then
        rust_ok=$(sed -n 's/^succeeded: *//p' "$run_dir/rust.log" 2>/dev/null || echo 0)
        rust_fail=$(sed -n 's/^failed: *//p' "$run_dir/rust.log" 2>/dev/null || echo 0)
    fi
    if [[ -f "$run_dir/python.log" ]]; then
        python_ok=$(sed -n 's/^succeeded: *//p' "$run_dir/python.log" 2>/dev/null || echo 0)
        python_fail=$(sed -n 's/^failed: *//p' "$run_dir/python.log" 2>/dev/null || echo 0)
    fi

    local has_review="no"
    [[ -f "$run_dir/review.md" ]] && has_review="yes"

    local diff_count=0
    if [[ -d "$run_dir/diffs" ]]; then
        diff_count=$(ls -1 "$run_dir/diffs/"*.diff 2>/dev/null | wc -l | tr -d ' ')
    fi

    printf "%s | %-10s | rust=%s/%s python=%s/%s | diffs=%s | review=%s | %s\n" \
        "$timestamp" "$doc_type" \
        "$rust_ok" "$((rust_ok + rust_fail))" \
        "$python_ok" "$((python_ok + python_fail))" \
        "$diff_count" "$has_review" \
        "$run_dir" \
        >> "$CENTRAL_LOG"

    ok "Central log updated: $CENTRAL_LOG"
}

# ── Commands ─────────────────────────────────────────────────────────────────

cmd_list() {
    for t in "${SUPPORTED_TYPES[@]}"; do
        local dir="$REAL_E2E_DIR/$t"
        if [[ -d "$dir" ]]; then
            local count
            count=$(ls -1 "$dir" 2>/dev/null | wc -l | tr -d ' ')
            echo -e "${BOLD}$t${NC} ($count files)"
            ls -1 "$dir" 2>/dev/null | while read -r f; do
                local size
                size=$(wc -c < "$dir/$f" | tr -d ' ')
                printf "  %-40s %s\n" "$f" "$(numfmt --to=iec-i --suffix=B $size 2>/dev/null || echo "${size} bytes")"
            done
            echo ""
        else
            echo -e "${BOLD}$t${NC} (not found)"
        fi
    done
}

cmd_run() {
    local doc_type="$1"
    local input_dir="$REAL_E2E_DIR/$doc_type"

    if [[ ! -d "$input_dir" ]]; then
        fatal "Input directory not found: $input_dir"
    fi

    local file_count
    file_count=$(ls -1 "$input_dir" 2>/dev/null | wc -l | tr -d ' ')
    if [[ "$file_count" -eq 0 ]]; then
        fatal "No files found in $input_dir"
    fi

    check_prerequisites

    local timestamp
    timestamp=$(date +"%Y%m%d_%H%M%S")
    local run_dir="$RUNS_DIR/$doc_type/$timestamp"
    mkdir -p "$run_dir/rust" "$run_dir/python" "$run_dir/diffs"

    log "Starting e2e comparison for ${BOLD}$doc_type${NC}"
    log "Input:  $input_dir ($file_count files)"
    log "Output: $run_dir"
    echo ""

    # ── Rust ──
    log "${BOLD}Running Rust (docling-rs)...${NC}"
    run_engine "rust" "$doc_type" "$input_dir" "$run_dir/rust" "$run_dir/rust.log"

    # ── Python ──
    log "${BOLD}Running Python (docling)...${NC}"
    run_engine "python" "$doc_type" "$input_dir" "$run_dir/python" "$run_dir/python.log"

    # ── Comparison report ──
    log "Generating comparison report..."
    generate_report "$run_dir" "$doc_type"

    # ── Cursor agent review ──
    if [[ -n "$SKIP_REVIEW" ]]; then
        warn "Skipping Cursor agent review (SKIP_REVIEW is set)"
    else
        launch_review "$run_dir" "$doc_type"
    fi

    # ── Central log ──
    append_central_log "$run_dir" "$doc_type" "$timestamp"

    echo ""
    log "${GREEN}${BOLD}Done!${NC} Results in: $run_dir"
    log "  comparison_report.md  — side-by-side summary"
    log "  diffs/                — per-file diffs"
    log "  review.md             — agent analysis (if available)"
}

# ── Main ─────────────────────────────────────────────────────────────────────

if [[ $# -lt 1 ]]; then
    usage
fi

# Handle "list" as first arg (no doc_type needed)
if [[ "$1" == "list" ]]; then
    cmd_list
    exit 0
fi

if [[ $# -lt 2 ]]; then
    usage
fi

DOC_TYPE="$1"
COMMAND="$2"

# Validate doc type
valid=false
for t in "${SUPPORTED_TYPES[@]}"; do
    if [[ "$t" == "$DOC_TYPE" ]]; then
        valid=true
        break
    fi
done
if [[ "$valid" == "false" ]]; then
    fatal "Unknown doc type: $DOC_TYPE\nSupported: ${SUPPORTED_TYPES[*]}"
fi

case "$COMMAND" in
    run) cmd_run "$DOC_TYPE" ;;
    *)   fatal "Unknown command: $COMMAND (expected: run)" ;;
esac
