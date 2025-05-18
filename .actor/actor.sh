#!/bin/bash

export PATH=$PATH:/build-files/node_modules/.bin

# Function to upload content to the key-value store
upload_to_kvs() {
    local content_file="$1"
    local key_name="$2"
    local content_type="$3"
    local description="$4"

    # Find the Apify CLI command
    find_apify_cmd
    local apify_cmd="$FOUND_APIFY_CMD"

    if [ -n "$apify_cmd" ]; then
        echo "Uploading $description to key-value store (key: $key_name)..."

        # Create a temporary home directory with write permissions
        setup_temp_environment

        # Use the --no-update-notifier flag if available
        if $apify_cmd --help | grep -q "\--no-update-notifier"; then
            if $apify_cmd --no-update-notifier actor:set-value "$key_name" --contentType "$content_type" < "$content_file"; then
                echo "Successfully uploaded $description to key-value store"
                local url="https://api.apify.com/v2/key-value-stores/${APIFY_DEFAULT_KEY_VALUE_STORE_ID}/records/$key_name"
                echo "$description available at: $url"
                cleanup_temp_environment
                return 0
            fi
        else
            # Fall back to regular command if flag isn't available
            if $apify_cmd actor:set-value "$key_name" --contentType "$content_type" < "$content_file"; then
                echo "Successfully uploaded $description to key-value store"
                local url="https://api.apify.com/v2/key-value-stores/${APIFY_DEFAULT_KEY_VALUE_STORE_ID}/records/$key_name"
                echo "$description available at: $url"
                cleanup_temp_environment
                return 0
            fi
        fi

        echo "ERROR: Failed to upload $description to key-value store"
        cleanup_temp_environment
        return 1
    else
        echo "ERROR: Apify CLI not found for $description upload"
        return 1
    fi
}

# Function to find Apify CLI command
find_apify_cmd() {
    FOUND_APIFY_CMD=""
    for cmd in "apify" "actor" "/usr/local/bin/apify" "/usr/bin/apify" "/opt/apify/cli/bin/apify"; do
        if command -v "$cmd" &> /dev/null; then
            FOUND_APIFY_CMD="$cmd"
            break
        fi
    done
}

# Function to set up temporary environment for Apify CLI
setup_temp_environment() {
    export TMPDIR="/tmp/apify-home-${RANDOM}"
    mkdir -p "$TMPDIR"
    export APIFY_DISABLE_VERSION_CHECK=1
    export NODE_OPTIONS="--no-warnings"
    export HOME="$TMPDIR"  # Override home directory to writable location
}

# Function to clean up temporary environment
cleanup_temp_environment() {
    rm -rf "$TMPDIR" 2>/dev/null || true
}

# Function to push data to Apify dataset
push_to_dataset() {
    # Example usage: push_to_dataset "$RESULT_URL" "$OUTPUT_SIZE" "zip"

    local result_url="$1"
    local size="$2"
    local format="$3"

    # Find Apify CLI command
    find_apify_cmd
    local apify_cmd="$FOUND_APIFY_CMD"

    if [ -n "$apify_cmd" ]; then
        echo "Adding record to dataset..."
        setup_temp_environment

        # Use the --no-update-notifier flag if available
        if $apify_cmd --help | grep -q "\--no-update-notifier"; then
            if $apify_cmd --no-update-notifier actor:push-data "{\"output_file\": \"${result_url}\", \"format\": \"${format}\", \"size\": \"${size}\", \"status\": \"success\"}"; then
                echo "Successfully added record to dataset"
            else
                echo "Warning: Failed to add record to dataset"
            fi
        else
            # Fall back to regular command
            if $apify_cmd actor:push-data "{\"output_file\": \"${result_url}\", \"format\": \"${format}\", \"size\": \"${size}\", \"status\": \"success\"}"; then
                echo "Successfully added record to dataset"
            else
                echo "Warning: Failed to add record to dataset"
            fi
        fi

        cleanup_temp_environment
    fi
}


# --- Setup logging and error handling ---

LOG_FILE="/tmp/docling.log"
touch "$LOG_FILE" || {
    echo "Fatal: Cannot create log file at $LOG_FILE"
    exit 1
}

# Log to both console and file
exec 1> >(tee -a "$LOG_FILE")
exec 2> >(tee -a "$LOG_FILE" >&2)

# Exit codes
readonly ERR_API_UNAVAILABLE=15
readonly ERR_INVALID_INPUT=16


# --- Debug environment ---

echo "Date: $(date)"
echo "Python version: $(python --version 2>&1)"
echo "Docling-serve path: $(which docling-serve 2>/dev/null || echo 'Not found')"
echo "Working directory: $(pwd)"

# --- Get input ---

echo "Getting Apify Actor Input"
INPUT=$(apify actor get-input 2>/dev/null)

# --- Setup tools ---

echo "Setting up tools..."
TOOLS_DIR="/tmp/docling-tools"
mkdir -p "$TOOLS_DIR"

# Copy tools if available
if [ -d "/build-files" ]; then
    echo "Copying tools from /build-files..."
    cp -r /build-files/* "$TOOLS_DIR/"
    export PATH="$TOOLS_DIR/bin:$PATH"
else
    echo "Warning: No build files directory found. Some tools may be unavailable."
fi

# Check OCR directories and ensure they're writable
echo "Checking OCR directory permissions..."
OCR_DIR="/opt/app-root/src/.EasyOCR"
if [ -d "$OCR_DIR" ]; then
    # Test if we can write to the directory
    if touch "$OCR_DIR/test_write" 2>/dev/null; then
        echo "[✓] OCR directory is writable"
        rm "$OCR_DIR/test_write"
    else
        echo "[✗] OCR directory is not writable, setting up alternative in /tmp"

        # Create alternative in /tmp (which is writable)
        mkdir -p "/tmp/.EasyOCR/user_network"
        export EASYOCR_MODULE_PATH="/tmp/.EasyOCR"
    fi
else
    echo "OCR directory not found, creating in /tmp"
    mkdir -p "/tmp/.EasyOCR/user_network"
    export EASYOCR_MODULE_PATH="/tmp/.EasyOCR"
fi


# --- Starting the API ---

echo "Starting docling-serve API..."

# Create a dedicated working directory in /tmp (writable)
API_DIR="/tmp/docling-api"
mkdir -p "$API_DIR"
cd "$API_DIR"
echo "API working directory: $(pwd)"

# Find docling-serve executable
DOCLING_SERVE_PATH=$(which docling-serve)
echo "Docling-serve executable: $DOCLING_SERVE_PATH"

# Start the API with minimal parameters to avoid any issues
echo "Starting docling-serve API..."
"$DOCLING_SERVE_PATH" run --host 0.0.0.0 --port 5001 > "$API_DIR/docling-serve.log" 2>&1 &
API_PID=$!
echo "Started docling-serve API with PID: $API_PID"

# A more reliable wait for API startup
echo "Waiting for API to initialize..."
MAX_TRIES=30
tries=0
started=false

while [ $tries -lt $MAX_TRIES ]; do
    tries=$((tries + 1))

    # Check if process is still running
    if ! ps -p $API_PID > /dev/null; then
        echo "ERROR: docling-serve API process terminated unexpectedly after $tries seconds"
        break
    fi

    # Check log for startup completion or errors
    if grep -q "Application startup complete" "$API_DIR/docling-serve.log" 2>/dev/null; then
        echo "[✓] API startup completed successfully after $tries seconds"
        started=true
        break
    fi

    if grep -q "Permission denied\|PermissionError" "$API_DIR/docling-serve.log" 2>/dev/null; then
        echo "ERROR: Permission errors detected in API startup"
        break
    fi

    # Sleep and check again
    sleep 1

    # Output a progress indicator every 5 seconds
    if [ $((tries % 5)) -eq 0 ]; then
        echo "Still waiting for API startup... ($tries/$MAX_TRIES seconds)"
    fi
done

# Show log content regardless of outcome
echo "docling-serve log output so far:"
tail -n 20 "$API_DIR/docling-serve.log"

# Verify the API is running
if ! ps -p $API_PID > /dev/null; then
    echo "ERROR: docling-serve API failed to start"
    if [ -f "$API_DIR/docling-serve.log" ]; then
        echo "Full log output:"
        cat "$API_DIR/docling-serve.log"
    fi
    exit $ERR_API_UNAVAILABLE
fi

if [ "$started" != "true" ]; then
    echo "WARNING: API process is running but startup completion was not detected"
    echo "Will attempt to continue anyway..."
fi

# Try to verify API is responding at this point
echo "Verifying API responsiveness..."
(python -c "
import sys, time, socket
for i in range(5):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1)
        result = s.connect_ex(('localhost', 5001))
        if result == 0:
            s.close()
            print('Port 5001 is open and accepting connections')
            sys.exit(0)
        s.close()
    except Exception as e:
        pass
    time.sleep(1)
print('Could not connect to API port after 5 attempts')
sys.exit(1)
" && echo "API verification succeeded") || echo "API verification failed, but continuing anyway"

# Define API endpoint
DOCLING_API_ENDPOINT="http://localhost:5001/v1alpha/convert/source"


# --- Processing document ---

echo "Starting document processing..."
echo "Reading input from Apify..."

echo "Input content:" >&2
echo "$INPUT" >&2  # Send the raw input to stderr for debugging
echo "$INPUT"      # Send the clean JSON to stdout for processing

# Create the request JSON

REQUEST_JSON=$(echo $INPUT | jq '.options += {"return_as_file": true}')

echo "Creating request JSON:" >&2
echo "$REQUEST_JSON" >&2
echo "$REQUEST_JSON" > "$API_DIR/request.json"


# Send the conversion request using our Python script
#echo "Sending conversion request to docling-serve API..."
#python "$TOOLS_DIR/docling_processor.py" \
#    --api-endpoint "$DOCLING_API_ENDPOINT" \
#    --request-json "$API_DIR/request.json" \
#    --output-dir "$API_DIR" \
#    --output-format "$OUTPUT_FORMAT"

echo "Curl the Docling API"
curl -s -H "content-type: application/json" -X POST --data-binary @$API_DIR/request.json -o $API_DIR/output.zip $DOCLING_API_ENDPOINT

CURL_EXIT_CODE=$?

# --- Check for various potential output files ---

echo "Checking for output files..."
if [ -f "$API_DIR/output.zip" ]; then
    echo "Conversion completed successfully! Output file found."

    # Get content from the converted file
    OUTPUT_SIZE=$(wc -c < "$API_DIR/output.zip")
    echo "Output file found with size: $OUTPUT_SIZE bytes"

    # Calculate the access URL for result display
    RESULT_URL="https://api.apify.com/v2/key-value-stores/${APIFY_DEFAULT_KEY_VALUE_STORE_ID}/records/OUTPUT"

    echo "=============================="
    echo "PROCESSING COMPLETE!"
    echo "Output size: ${OUTPUT_SIZE} bytes"
    echo "=============================="

    # Set the output content type based on format
    CONTENT_TYPE="application/zip"

    # Upload the document content using our function
    upload_to_kvs "$API_DIR/output.zip" "OUTPUT" "$CONTENT_TYPE" "Document content"

    # Only proceed with dataset record if document upload succeeded
    if [ $? -eq 0 ]; then
        echo "Your document is available at: ${RESULT_URL}"
        echo "=============================="

        # Push data to dataset
        push_to_dataset "$RESULT_URL" "$OUTPUT_SIZE" "zip"
    fi
else
    echo "ERROR: No converted output file found at $API_DIR/output.zip"

    # Create error metadata
    ERROR_METADATA="{\"status\":\"error\",\"error\":\"No converted output file found\",\"documentUrl\":\"$DOCUMENT_URL\"}"
    echo "$ERROR_METADATA" > "/tmp/actor-output/OUTPUT"
    chmod 644 "/tmp/actor-output/OUTPUT"

    echo "Error information has been saved to /tmp/actor-output/OUTPUT"
fi


# --- Verify output files for debugging ---

echo "=== Final Output Verification ==="
echo "Files in /tmp/actor-output:"
ls -la /tmp/actor-output/ 2>/dev/null || echo "Cannot list /tmp/actor-output/"

echo "All operations completed. The output should be available in the default key-value store."
echo "Content URL: ${RESULT_URL:-No URL available}"


# --- Cleanup function ---

cleanup() {
    echo "Running cleanup..."

    # Stop the API process
    if [ -n "$API_PID" ]; then
        echo "Stopping docling-serve API (PID: $API_PID)..."
        kill $API_PID 2>/dev/null || true
    fi

    # Export log file to KVS if it exists
    # DO THIS BEFORE REMOVING TOOLS DIRECTORY
    if [ -f "$LOG_FILE" ]; then
        if [ -s "$LOG_FILE" ]; then
            echo "Log file is not empty, pushing to key-value store (key: LOG)..."

            # Upload log using our function
            upload_to_kvs "$LOG_FILE" "LOG" "text/plain" "Log file"
        else
            echo "Warning: log file exists but is empty"
        fi
    else
        echo "Warning: No log file found"
    fi

    # Clean up temporary files AFTER log is uploaded
    echo "Cleaning up temporary files..."
    if [ -d "$API_DIR" ]; then
        echo "Removing API working directory: $API_DIR"
        rm -rf "$API_DIR" 2>/dev/null || echo "Warning: Failed to remove $API_DIR"
    fi

    if [ -d "$TOOLS_DIR" ]; then
        echo "Removing tools directory: $TOOLS_DIR"
        rm -rf "$TOOLS_DIR" 2>/dev/null || echo "Warning: Failed to remove $TOOLS_DIR"
    fi

    # Keep log file until the very end
    echo "Script execution completed at $(date)"
    echo "Actor execution completed"
}

# Register cleanup
trap cleanup EXIT
