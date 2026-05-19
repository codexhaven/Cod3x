
#!/bin/bash
# ctx: codexhaven

# Machine-Centric Neural Lingua (MCNL) Orchestrator
# Automates the build, validation, and lifecycle management of MCNL modules.

set -euo pipefail

# Configuration: Use absolute paths for robust resolution
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCHEMA_FILE="$PROJECT_ROOT/mcnl.fbs"
RUNTIME_SRC="$PROJECT_ROOT/runtime.cpp"
VALIDATOR_SRC="$PROJECT_ROOT/validator.cpp"
BUILD_DIR="$PROJECT_ROOT/build"
LOG_FILE="$PROJECT_ROOT/orchestrator.log"

# Log messages with consistent format
# Input: Message string
log() {
    echo "[$(date +'%Y-%m-%dT%H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Verify build dependencies and create directories
# Exits on failure
init_build() {
    log "Initializing build environment..."
    mkdir -p "$BUILD_DIR"
    
    if [ ! -f "$SCHEMA_FILE" ]; then
        log "Error: Missing schema file $SCHEMA_FILE"
        exit 1
    fi
    
    if [ ! -f "$RUNTIME_SRC" ] || [ ! -f "$VALIDATOR_SRC" ]; then
        log "Error: Missing source files in $PROJECT_ROOT"
        exit 1
    fi
}

# Compile the C++ Runtime Shim
# Requires clang++ and llama.cpp development headers
compile_runtime() {
    log "Compiling Runtime Shim..."
    if ! clang++ -O3 -std=c++17 "$RUNTIME_SRC" -o "$BUILD_DIR/mcnl_runtime" \
        -I/usr/local/include/llama.cpp -L/usr/local/lib -lllama; then
        log "Error: Runtime compilation failed."
        exit 1
    fi
    log "Runtime compiled successfully."
}

# Compile the Validator Engine
# Exits on failure
compile_validator() {
    log "Compiling Validator Engine..."
    if ! clang++ -O3 -std=c++17 "$VALIDATOR_SRC" -o "$BUILD_DIR/mcnl_validator"; then
        log "Error: Validator compilation failed."
        exit 1
    fi
    log "Validator compiled successfully."
}

# Execute the MCNL lifecycle loop
# Input: Path to binary payload
run_cycle() {
    local payload="$1"
    
    if [ -z "$payload" ]; then
        log "Error: No payload provided for run_cycle."
        exit 1
    fi
    
    if [ ! -f "$payload" ]; then
        log "Error: Payload file not found: $payload"
        exit 1
    fi

    log "Executing MCNL Neural Loop for $payload..."
    
    if ! "$BUILD_DIR/mcnl_runtime" --payload "$payload"; then
        log "Error: Execution failed during runtime injection."
        exit 1
    fi
    
    log "Payload injected. Validating..."
    if ! "$BUILD_DIR/mcnl_validator" --check; then
        log "Error: Validation failed post-injection."
        exit 1
    fi
    
    log "Cycle completed successfully."
}

# Primary entry point
# Handles initialization and command execution
main() {
    case "${1:-}" in
        --init)
            init_build
            compile_runtime
            compile_validator
            ;;
        --run)
            if [ -z "${2:-}" ]; then
                log "Error: --run requires a payload binary file."
                exit 1
            fi
            run_cycle "$2"
            ;;
        *)
            echo "Usage: $0 {--init|--run <payload_bin>}"
            exit 1
            ;;
    esac
}

main "$@"