#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Check if a directory is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

INPUT_DIR="$1"

# Get the absolute path for reliable relative path calculation later
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

echo "Input Directory: $INPUT_DIR_ABS"

declare -A processed_files # Associative array to store original -> hipified mappings

# Process each .cu and .cpp file in the directory recursively
echo "Starting Hipify process..."

# Use process substitution and mapfile for efficiency if available (Bash 4+)
# Otherwise, use a temporary file or loop as before. Sticking to original loop structure for compatibility.
find "$INPUT_DIR_ABS" -type f \( -name "*.cu" -o -name "*.cpp" -o -name "*.hpp" -o -name "*.cuh" \) | while read -r file; do
    # Skip files already ending in _hipified.*
    if [[ "$file" == *_hipified.* ]]; then
        echo "Skipping already hipified file: $file"
        continue
    fi

    # --- Determine Output Filename ---
    base_name="${file%.*}"  # Get path and filename without extension
    extension="${file##*.}" # Get the original extension

    output_extension="" # Variable to hold the target extension

    # Set the output extension based on the input extension
    case "$extension" in
        cu)
            output_extension="cpp" # Change .cu input to .cpp output
            ;;
        cpp)
            output_extension="cpp" # Keep .cpp as .cpp
            ;;
        hpp)
            output_extension="hpp" # Keep .hpp as .hpp
            ;;
        cuh)
            output_extension="cuh" # Keep .cuh as .cuh
            ;;
        *)
            # This case should not be reached due to the 'find' command filters,
            # but it's good practice to handle unexpected scenarios.
            echo "Warning: Skipping file with unexpected extension '$extension': $file" >&2
            continue # Skip to the next file
            ;;
        esac

    OUTPUT_FILE="${base_name}_hipified.${output_extension}"
    # --- End Output Filename Determination ---

    echo "Processing: $file -> $OUTPUT_FILE"


    # Run hipify-perl and check exit status
    if hipify-perl "$file" > "$OUTPUT_FILE"; then
        # Store relative paths for CMake processing later
        original_relative="${file#$INPUT_DIR_ABS/}"
        hipified_relative="${OUTPUT_FILE#$INPUT_DIR_ABS/}"
        # Using a temporary marker file approach as bash subshells prevent direct array modification
        # A more robust method might use process substitution or temp files explicitly.
        # For simplicity here, we'll re-find the files later but use relative paths.
        echo "Successfully processed: $file"
    else
        echo "Error processing $file with hipify-perl. Exit status: $?. Removing incomplete output: $OUTPUT_FILE"
        rm -f "$OUTPUT_FILE" # Clean up partial output
        # Decide whether to continue or exit on error
        # exit 1 # Uncomment to stop on first error
    fi
done

echo "Hipify process completed."

# Update CMakeLists.txt
# Look for CMakeLists.txt recursively as well? For now, stick to the original assumption.
CMAKE_FILE="$INPUT_DIR_ABS/CMakeLists.txt"

if [ -f "$CMAKE_FILE" ]; then
    echo "Attempting to update CMakeLists.txt: $CMAKE_FILE"

    # Create a backup
    cp "$CMAKE_FILE" "${CMAKE_FILE}.bak"
    echo "Backup created: ${CMAKE_FILE}.bak"

    # Replace nvcc with hipcc (still a simple replacement, manual review recommended)
    # Use a temporary file for sed operations to avoid issues with multiple -i calls potentially
    TMP_CMAKE=$(mktemp)
    sed 's/nvcc/hipcc/g' "$CMAKE_FILE" > "$TMP_CMAKE"
    mv "$TMP_CMAKE" "$CMAKE_FILE"
    echo "Replaced 'nvcc' with 'hipcc' (basic string substitution)."

    echo "Updating source filenames in CMakeLists.txt..."
    # Re-find the original files to determine pairs (less efficient, but avoids complexity of passing data from subshell)
    find "$INPUT_DIR_ABS" -type f \( -name "*.cu" -o -name "*.cpp" \) | while read -r original_file_abs; do
         # Skip hipified files in this find pass
        if [[ "$original_file_abs" == *_hipified.* ]]; then
            continue
        fi

        hipified_file_abs="${original_file_abs%.*}_hipified.${original_file_abs##*.}"

        # Check if the corresponding hipified file actually exists (was successfully created)
        if [ -f "$hipified_file_abs" ]; then
            # Calculate paths relative to the CMakeLists.txt directory (INPUT_DIR_ABS)
            original_relative="${original_file_abs#$INPUT_DIR_ABS/}"
            hipified_relative="${hipified_file_abs#$INPUT_DIR_ABS/}"

            # Escape paths for sed (especially slashes and other special chars)
            # Basic escaping for / - might need more for complex paths
            original_escaped=$(echo "$original_relative" | sed 's/[\/&]/\\&/g')
            hipified_escaped=$(echo "$hipified_relative" | sed 's/[\/&]/\\&/g')

            # Use sed to replace the relative paths. Use a distinct delimiter.
            # The pattern tries to match the path possibly surrounded by spaces, quotes, parens, or line ends/starts
            # This is still heuristic and might not catch all CMake syntax variations.
            TMP_CMAKE=$(mktemp)
            sed "s/\([\"'( ]\|^\)${original_escaped}\([\"') ]\|$\)/\1${hipified_escaped}\2/g" "$CMAKE_FILE" > "$TMP_CMAKE"
            # Check if sed actually changed anything to avoid unnecessary moves/potential timestamp changes
            if ! cmp -s "$CMAKE_FILE" "$TMP_CMAKE"; then
                 echo "Replacing '$original_relative' with '$hipified_relative'"
                 mv "$TMP_CMAKE" "$CMAKE_FILE"
            else
                 rm "$TMP_CMAKE" # No change, remove temp file
            fi

        fi
    done

    echo "CMakeLists.txt update attempt finished. Please review the changes manually: $CMAKE_FILE"
else
    echo "No CMakeLists.txt found directly in $INPUT_DIR_ABS. Skipping CMake update."
fi

echo "Script finished."