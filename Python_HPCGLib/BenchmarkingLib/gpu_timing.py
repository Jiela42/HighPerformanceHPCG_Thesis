import os
import torch
import datetime

class gpu_timer:

    def __init__(self, version_name, ault_node, matrix_type, nx, ny, nz, nnz, folder_path):
        self.folder_path = folder_path
        self.version_name = version_name
        self.ault_node = ault_node
        self.matrix_type = matrix_type
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.nnz = nnz        

        self.start_event = None
        self.end_event = None

        self.elapsed_CG_time_ms = []
        self.CG_time = False

        self.elapsed_MG_time_ms = []
        self.MG_time = False

        self.elapsed_SymGS_time_ms = []
        self.SymGS_time = False

        self.elapsed_SPMV_time_ms = []
        self.SPMV_time = False

        self.elapsed_waxpby_time_ms = []
        self.waxpby_time = False

        self.elapsed_dot_time_ms = []
        self.dot_time = False

    def start_timer(self):

        if self.start_event is not None:
            raise ValueError("Timer already started")
        
        torch.cuda.synchronize()

        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.start_event.record()

    def stop_timer(self, method_name):

        if self.start_event is None:
            raise ValueError("Timer not started")
        
        self.end_event.record()
        torch.cuda.synchronize()

        elapsed_time = self.start_event.elapsed_time(self.end_event)

        if method_name == "computeCG":
            self.CG_time = True
            self.elapsed_CG_time_ms.append(elapsed_time)
        elif method_name == "computeMG":
            self.MG_time = True
            self.elapsed_MG_time_ms.append(elapsed_time)
        elif method_name == "computeSymGS":
            self.SymGS_time = True
            self.elapsed_SymGS_time_ms.append(elapsed_time)
        elif method_name == "computeSPMV":
            self.SPMV_time = True
            self.elapsed_SPMV_time_ms.append(elapsed_time)
        elif method_name == "computeWAXPBY":
            self.waxpby_time = True
            self.elapsed_waxpby_time_ms.append(elapsed_time)
        elif method_name == "computeDot":
            self.dot_time = True
            self.elapsed_dot_time_ms.append(elapsed_time)
        else:
            raise ValueError("Method name not recognized")

        self.start_event = None
        self.end_event = None
        

    def destroy_timer(self):


        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)

        # write the results to a file
        if self.CG_time:
            filename = f"{self.folder_path}/{self.version_name}_{self.ault_node}_{self.matrix_type}_{self.nx}x{self.ny}x{self.ny}_CG.csv"
            with open(filename, "w") as f:
                f.write(f"{self.version_name},{self.ault_node},{self.matrix_type},{self.nx},{self.ny},{self.ny},{self.nnz},CG\n")
                for time in self.elapsed_CG_time_ms:
                    f.write(f"{time}\n")
        if self.MG_time:
            filename = f"{self.folder_path}/{self.version_name}_{self.ault_node}_{self.matrix_type}_{self.nx}x{self.ny}x{self.ny}_MG.csv"
            with open(filename, "w") as f:
                f.write(f"{self.version_name},{self.ault_node},{self.matrix_type},{self.nx},{self.ny},{self.ny},{self.nnz},MG\n")
                for time in self.elapsed_MG_time_ms:
                    f.write(f"{time}\n")
        if self.SymGS_time:
            filename = f"{self.folder_path}/{self.version_name}_{self.ault_node}_{self.matrix_type}_{self.nx}x{self.ny}x{self.ny}_SymGS.csv"
            with open(filename, "w") as f:
                f.write(f"{self.version_name},{self.ault_node},{self.matrix_type},{self.nx},{self.ny},{self.ny},{self.nnz},SymGS\n")
                for time in self.elapsed_SymGS_time_ms:
                    f.write(f"{time}\n")
        if self.SPMV_time:
            filename = f"{self.folder_path}/{self.version_name}_{self.ault_node}_{self.matrix_type}_{self.nx}x{self.ny}x{self.ny}_SPMV.csv"
            with open(filename, "w") as f:
                f.write(f"{self.version_name},{self.ault_node},{self.matrix_type},{self.nx},{self.ny},{self.ny},{self.nnz},SPMV\n")
                for time in self.elapsed_SPMV_time_ms:
                    f.write(f"{time}\n")
        if self.waxpby_time:
            filename = f"{self.folder_path}/{self.version_name}_{self.ault_node}_{self.matrix_type}_{self.nx}x{self.ny}x{self.ny}_WAXPBY.csv"
            with open(filename, "w") as f:
                    f.write(f"{self.version_name},{self.ault_node},{self.matrix_type},{self.nx},{self.ny},{self.ny},{self.nnz},WAXPBY\n")
                    for time in self.elapsed_waxpby_time_ms:
                        f.write(f"{time}\n")
        if self.dot_time:
            filename = f"{self.folder_path}/{self.version_name}_{self.ault_node}_{self.matrix_type}_{self.nx}x{self.ny}x{self.ny}_Dot.csv"
            with open(filename, "w") as f:
                    f.write(f"{self.version_name},{self.ault_node},{self.matrix_type},{self.nx},{self.ny},{self.ny},{self.nnz},Dot\n")
                    for time in self.elapsed_dot_time_ms:
                        f.write(f"{time}\n")
        
        # print("Timer destroyed")
        
            
