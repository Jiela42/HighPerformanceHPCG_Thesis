
import sys
import os


#  readin the textfile
filename = "debug_output1.txt"

with open(filename, 'r') as f:
    lines = f.readlines()

max_written_line = -1
max_line_bid = -1
max_line_tid = -1
debug_info = []

for line in lines:
    line = line.strip()

    if line.startswith("writing row: "):
        written_line = int(line.split()[2])
        bid = int(line.split()[5])
        tid = int(line.split()[9])
        if written_line > max_written_line:
            max_written_line = written_line
            max_line_bid = bid
            max_line_tid = tid
    else:
        debug_info.append(line)


print("Max written line: ", max_written_line)
print("Block ID: ", max_line_bid)
print("Thread ID: ", max_line_tid)
print("Debug info: ")
for line in debug_info:
    print(line)