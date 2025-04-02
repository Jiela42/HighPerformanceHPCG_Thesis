import os
import re
import sys

def hipify_shfl_down_sync(cu_file_path):
    """
    Modifies a .cu file to convert __shfl_down_sync(0xFFFFFFFF, my_sum, offset)
    to __shfl_down_sync(static_cast<unsigned long long> 0xFFFFFFFFFFFFFFFF, my_sum, offset).
    """
    try:
        with open(cu_file_path, 'r') as f:
            content = f.read()

        pattern = r'__shfl_down_sync\(0xFFFFFFFF,\s*([^,]+),\s*([^)]+)\)'
        replacement = r'__shfl_down_sync(static_cast<unsigned long long> 0xFFFFFFFFFFFFFFFF, \1, \2)'

        modified_content = re.sub(pattern, replacement, content)

        with open(cu_file_path, 'w') as f:
            f.write(modified_content)

        print(f"Modified: {cu_file_path}")

    except FileNotFoundError:
        print(f"Error: File not found at {cu_file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

def process_directory_recursive(directory):
    """
    Recursively processes all .cu files in the given directory and its subdirectories.
    """
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".cu"):
                file_path = os.path.join(root, file)
                hipify_shfl_down_sync(file_path)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python hipify_script.py <directory_path>")
        sys.exit(1)

    directory_path = sys.argv[1]

    if not os.path.isdir(directory_path):
        print(f"Error: '{directory_path}' is not a valid directory.")
        sys.exit(1)

    process_directory_recursive(directory_path)
