import os
import sys
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from HighPerformanceHPCG_Thesis.Python_HPCGLib.MatrixLib import generations



def print_l2_y(nx: int, ny: int, nz: int):
    y = generations.generate_y_forHPCG_problem(nx, ny, nz)
    
    # get norm directly on GPU
    l2_y = torch.norm(y).item()
    print(f"\"{nx}x{ny}x{nz}\": {l2_y}", flush=True)

if __name__ == "__main__":
    print("Running print_l2_y")
    # print_l2_y(8,8,8)
    # print_l2_y(16,16,16)
    # print_l2_y(24,24,24)
    # print_l2_y(32,32,32)
    # print_l2_y(64,64,64)
    # print_l2_y(128,64,64)
    # print_l2_y(128,128,64)
    print_l2_y(128,128,128)
    print_l2_y(256,128,128)
