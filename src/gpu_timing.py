import torch

class gpu_timer:

    

    def __init__(self, version_name, ault_node, matrix_type, nx, ny, nz):
        self.version_name = version_name
        self.ault_node = ault_node
        self.matrix_type = matrix_type
        self.nx = nx
        self.ny = ny
        self.nz = nz        

        self.CG_start_event = None
        self.CG_end_event = None
        self.elapsed_CG_time_ms = []
        self.CG_time = False


    def start_CG_timer(self):

        if self.CG_start_event is not None:
            raise ValueError("Timer already started")

        self.CG_time = True

        self.CG_start_event = torch.cuda.Event(enable_timing=True)
        self.CG_end_event = torch.cuda.Event(enable_timing=True)
        self.CG_start_event.record()

    def stop_CG_timer(self):
        
        if self.CG_start_event is None:
            raise ValueError("Timer not started")
        
        self.CG_end_event.record()
        torch.cuda.synchronize()

        self.elapsed_CG_time_ms.append(self.CG_start_event.elapsed_time(self.CG_end_event))

        self.CG_start_event = None
        self.CG_end_event = None

    def destroy_timers(self):

        print("Destroying timers")
        # write the results to a file
        # this needs to be reworked
        if self.CG_time:
            with open(f"../data/{self.version_name}_{self.ault_node}_{self.matrix_type}_CG_times.txt", "w") as f:
                for time in self.elapsed_CG_time_ms:
                    f.write(f"{time}\n")
