import os
import re
import sys

def update_includes(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)

            # Process only .cpp and .cu files
            if file.endswith("_hipified.cpp") or file.endswith("_hipified.cu") or file.endswith("_hipified.hpp") or file.endswith("_hipified.cuh"):
                with open(file_path, "r") as f:
                    content = f.read()

                # Replace includes like #include "file.cpp" -> #include "file_hipified.cpp"
                # And #include "file.cu" -> #include "file_hipified.cu"
                updated_content = re.sub(
                    r'#include\s+"([\w/]+)\.(cpp|cu|hpp|cuh)"',
                    r'#include "\1_hipified.\2"',
                    content
                )

                # Write back only if changes were made
                if content != updated_content:
                    with open(file_path, "w") as f:
                        f.write(updated_content)
                    print(f"Updated includes in: {file_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python update_includes.py <directory>")
        sys.exit(1)

    directory = sys.argv[1]

    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        sys.exit(1)

    update_includes(directory)
