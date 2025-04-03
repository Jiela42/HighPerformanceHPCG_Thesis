import os
import sys

def delete_non_hipified_files(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            # Check for .cpp and .cu files that do not have _hipified in their name
            if (file.endswith(".cpp") or file.endswith(".cu")) and "_hipified" not in file:
                file_path = os.path.join(root, file)
                os.remove(file_path)
                print(f"Deleted: {file_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python delete_non_hipified.py <directory>")
        sys.exit(1)

    directory = sys.argv[1]

    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        sys.exit(1)

    delete_non_hipified_files(directory)
