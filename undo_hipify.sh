#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
# Set to true to skip confirmations for file deletion and backup restoration
AUTO_CONFIRM_DELETE=false
AUTO_CONFIRM_RESTORE=false
# Set to true to attempt programmatic CMake reversal if backup not restored (USE WITH CAUTION)
ATTEMPT_PROGRAMMATIC_REVERSAL=false

# --- Input Validation ---
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <directory>"
    echo "Description: Attempts to undo the hipify process performed by the counterpart script."
    echo "  - Deletes *_hipified.* files."
    echo "  - Restores CMakeLists.txt from backup (.bak) if found and confirmed."
    echo "  - Optionally attempts programmatic reversal of CMake changes if no backup is restored."
    exit 1
fi

INPUT_DIR="$1"

# Check if realpath is available
if ! command -v realpath &> /dev/null; then
    echo "Error: 'realpath' command not found. Please install it (e.g., sudo apt install coreutils or brew install coreutils)."
    exit 1
fi
INPUT_DIR_ABS=$(realpath "$INPUT_DIR")


# Check if the directory exists
if [ ! -d "$INPUT_DIR_ABS" ]; then
    echo "Error: Directory '$INPUT_DIR_ABS' not found!"
    exit 1
fi

echo "Target Directory for Undo Operation: $INPUT_DIR_ABS"

# --- CMake Restoration ---
CMAKE_FILE="$INPUT_DIR_ABS/CMakeLists.txt"
CMAKE_BACKUP="${CMAKE_FILE}.bak"
cmake_needs_programmatic_reversal=false

if [ -f "$CMAKE_BACKUP" ]; then
    confirm_restore="n"
    if [ "$AUTO_CONFIRM_RESTORE" = true ]; then
        confirm_restore="y"
        echo "Auto-confirming CMake restore from backup."
    else
        read -p "Found backup: $CMAKE_BACKUP. Restore it (this is the recommended undo)? (y/N): " confirm_restore
    fi

    if [[ "$confirm_restore" =~ ^[Yy]$ ]]; then
        echo "Restoring CMakeLists.txt from backup..."
        # Move backup to original, overwriting current CMakeLists.txt
        mv "$CMAKE_BACKUP" "$CMAKE_FILE"
        echo "Restored '$CMAKE_FILE' from '$CMAKE_BACKUP'. The backup file is now gone."
        # If restored from backup, no need for programmatic reversal
        cmake_needs_programmatic_reversal=false
    else
        echo "Skipping restore from backup '$CMAKE_BACKUP'."
        # If backup exists but wasn't restored, check if user wants programmatic reversal
        if [ "$ATTEMPT_PROGRAMMATIC_REVERSAL" = true ] && [ -f "$CMAKE_FILE" ]; then
             cmake_needs_programmatic_reversal=true
             echo "Will attempt programmatic reversal since backup was not restored."
        fi
    fi
elif [ -f "$CMAKE_FILE" ]; then
    echo "No CMakeLists.txt.bak backup found."
    if [ "$ATTEMPT_PROGRAMMATIC_REVERSAL" = true ]; then
        cmake_needs_programmatic_reversal=true
        echo "Will attempt programmatic reversal as configured."
    else
        echo "No backup found and programmatic reversal is disabled. CMakeLists.txt will not be changed."
    fi
else
    echo "No CMakeLists.txt or CMakeLists.txt.bak found. Skipping CMake modifications."
fi

# --- Programmatic CMake Reversal (Conditional) ---
if [ "$cmake_needs_programmatic_reversal" = true ]; then
    echo "---"
    echo "Attempting Programmatic CMake Reversal (USE WITH CAUTION)"
    echo "This attempts to reverse changes if the backup wasn't used."
    echo "Manual review of CMakeLists.txt is highly recommended afterwards."

    # Create a backup *before* attempting programmatic reversal
    UNDO_BACKUP="${CMAKE_FILE}.undo.bak"
    cp "$CMAKE_FILE" "$UNDO_BACKUP"
    echo "Backup of current state saved to '$UNDO_BACKUP'"

    # 1. Reverse hipcc -> nvcc
    echo "Reverting 'hipcc' to 'nvcc' (basic string substitution)..."
    # Use temp file for atomicity
    TMP_CMAKE_REVERT=$(mktemp)
    sed 's/hipcc/nvcc/g' "$CMAKE_FILE" > "$TMP_CMAKE_REVERT"
    if ! cmp -s "$CMAKE_FILE" "$TMP_CMAKE_REVERT"; then
        mv "$TMP_CMAKE_REVERT" "$CMAKE_FILE"
        echo "Replaced 'hipcc' with 'nvcc'."
    else
        rm "$TMP_CMAKE_REVERT"
        echo "'hipcc' not found or no changes made."
    fi

    # 2. Reverse filenames (_hipified.* -> .*)
    echo "Searching for hipified files to revert their names in CMakeLists.txt..."
    # Need to find the hipified files first to know what to revert
    find "$INPUT_DIR_ABS" -type f \( -name "*_hipified.cu" -o -name "*_hipified.cpp" \) | while read -r hipified_file_abs; do
        # Infer original filename - CAVEAT: Breaks if original filename contained '_hipified'
        original_file_abs=$(echo "$hipified_file_abs" | sed 's/_hipified//')

        # Calculate relative paths from the CMakeLists.txt directory
        hipified_relative="${hipified_file_abs#$INPUT_DIR_ABS/}"
        original_relative="${original_file_abs#$INPUT_DIR_ABS/}"

        # Ensure paths are not empty and look reasonable
        if [ -z "$hipified_relative" ] || [ -z "$original_relative" ]; then
            echo "Warning: Could not determine relative paths for '$hipified_file_abs'. Skipping CMake revert for this file."
            continue
        fi

        # Escape paths for sed
        hipified_escaped=$(echo "$hipified_relative" | sed 's/[\/&]/\\&/g')
        original_escaped=$(echo "$original_relative" | sed 's/[\/&]/\\&/g')

        # Use sed to replace the relative paths (inverse of the hipify script)
        TMP_CMAKE_REVERT=$(mktemp)
        # Using the same pattern structure as the improved hipify script
        sed "s/\([\"'( ]\|^\)${hipified_escaped}\([\"') ]\|$\)/\1${original_escaped}\2/g" "$CMAKE_FILE" > "$TMP_CMAKE_REVERT"

        # Check if sed actually changed anything
        if ! cmp -s "$CMAKE_FILE" "$TMP_CMAKE_REVERT"; then
             echo "Reverting '$hipified_relative' to '$original_relative' in CMakeLists.txt"
             mv "$TMP_CMAKE_REVERT" "$CMAKE_FILE"
        else
             rm "$TMP_CMAKE_REVERT" # No change, remove temp file
        fi
    done
    echo "Programmatic CMake reversal attempt finished. PLEASE REVIEW '$CMAKE_FILE' MANUALLY."
    echo "---"
fi


# --- Delete Hipified Files ---
echo "Searching for *_hipified.cu and *_hipified.cpp files to delete..."
# Use mapfile (Bash 4+) to store filenames safely, handling spaces etc.
mapfile -t hipified_files < <(find "$INPUT_DIR_ABS" -depth -type f \( -name "*_hipified.cu" -o -name "*_hipified.cpp" \))
# -depth helps if you have nested directories like foo_hipified/bar.cpp_hipified but is generally safe

if [ ${#hipified_files[@]} -eq 0 ]; then
    echo "No *_hipified.* files found to delete."
else
    echo "Found the following ${#hipified_files[@]} hipified files:"
    printf "  %s\n" "${hipified_files[@]}"

    confirm_delete="n"
    if [ "$AUTO_CONFIRM_DELETE" = true ]; then
        confirm_delete="y"
        echo "Auto-confirming deletion."
    else
         read -p "Delete these ${#hipified_files[@]} files? (y/N): " confirm_delete
    fi

    if [[ "$confirm_delete" =~ ^[Yy]$ ]]; then
        echo "Deleting files..."
        # Delete files listed in the array. Using printf and xargs is robust.
        # -r ensures xargs doesn't run rm if the list is empty.
        # -0 with find/xargs is safer for weird filenames, but mapfile handles most cases here.
        printf "%s\n" "${hipified_files[@]}" | xargs -r rm -v
        echo "Deletion complete."
    else
        echo "Skipping deletion of hipified files."
    fi
fi

echo "Undo script finished."